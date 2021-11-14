from sklearn.metrics import f1_score
from torch import nn
from itertools import cycle
import logging
import os
import numpy as np
import time
import torch
from torch.optim.lr_scheduler import CyclicLR
from torchcontrib.optim import SWA
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from f1_opt import OptimizedF1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Running:
    def __init__(self, model, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.result = None  # 日志写入
        self.best_acc = 0.0  # 保存训练期间的最优精度
        self.best_coef = []
        self.model = model
        # 数据并行
        self.gpu_num = torch.cuda.device_count()
        if self.gpu_num > 1:
            logger.info("Model has been loaded, using {} GPUs!".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)
        else:
            logger.info('Model has been loaded, using {}!'.format(self.device))
        self.model.to(self.device)

    def train(self, train_loader, eval_loader, epoch):
        # 定义优化器
        if self.args.swa:
            logger.info('SWA training')
            base_opt = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            optimizer = SWA(base_opt, swa_start=self.args.swa_start, swa_freq=self.args.swa_freq,
                            swa_lr=self.args.swa_lr)
            scheduler = CyclicLR(optimizer, base_lr=5e-5, max_lr=7e-5,
                                 step_size_up=(self.args.epochs * len(train_loader) / self.args.batch_accumulation),
                                 cycle_momentum=False)
        else:
            logger.info('Adam training')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup,
                                                        num_training_steps=(self.args.epochs * len(
                                                            train_loader) / self.args.batch_accumulation))

        bar = tqdm(range(self.args.train_steps), total=self.args.train_steps)
        train_batches = cycle(train_loader)
        loss_sum = 0.0
        start = time.time()
        self.model.train()
        for step in bar:
            batch = next(train_batches)
            input_ids, input_mask, segment_ids, label_ids = [t.to(self.device) for t in batch]

            loss, _ = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                 labels=label_ids)
            if self.gpu_num > 1:
                loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # optimizer.update_swa()
            loss_sum += loss.cpu().item()
            train_loss = loss_sum / (step + 1)

            bar.set_description("loss {}".format(train_loss))
            if (step + 1) % self.args.eval_steps == 0:
                logger.info("***** Training result *****")
                logger.info('  time %.2fs ', time.time() - start)
                logger.info("  %s = %s", 'global_step', str(step + 1))
                logger.info("  %s = %s", 'train loss', str(train_loss))
                # 每eval_steps进行一次evaluate
                self.result = {'epoch': epoch, 'global_step': step + 1, 'loss': train_loss}
                if self.args.swa:
                    optimizer.swap_swa_sgd()
                self.evaluate(eval_loader, epoch)
                if self.args.swa:
                    optimizer.swap_swa_sgd()
        if self.args.swa:
            optimizer.swap_swa_sgd()
        logging.info('The training  of epoch ' + str(epoch + 1) + ' has finished.')

    def compute_accuracy(self, out, labels):
        #         outputs = np.argmax(out, axis=1)

        op = OptimizedF1()
        op.fit(out, labels)
        coef = op.coefficients()
        outputs = coef * out
        outputs = np.argmax(outputs, axis=1)
        return f1_score(labels, outputs, labels=[0, 1, 2], average='macro'),coef

    def evaluate(self, eval_loader, epoch):
        self.model.eval()
        inference_labels = []
        real_labels = []
        eval_loss_sum, eval_accuracy = 0, 0
        eval_steps = 0
        for input_ids, input_mask, segment_ids, label_ids in eval_loader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)

            with torch.no_grad():
                eval_loss, logits = self.model(input_ids=input_ids, token_type_ids=segment_ids,
                                               attention_mask=input_mask, labels=label_ids)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            inference_labels.append(logits)
            real_labels.append(label_ids)
            if self.gpu_num > 1:
                eval_loss = eval_loss.mean()
            eval_loss_sum += eval_loss.cpu().item()
            eval_steps += 1

        real_labels = np.concatenate(real_labels, 0)
        inference_labels = np.concatenate(inference_labels, 0).reshape(-1, self.args.num_classes)
        self.model.train()
        eval_loss = eval_loss_sum / eval_steps
        eval_accuracy,eval_coef = self.compute_accuracy(inference_labels, real_labels)
        self.result['eval_loss'] = eval_loss
        self.result['eval_accuracy'] = eval_accuracy

        # 保存日志和模型
        output_path = self.args.output_path + str(epoch)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_eval_file = os.path.join(output_path, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            for key in sorted(self.result.keys()):
                logger.info("  %s = %s", key, str(self.result[key]))
                writer.write("%s = %s\n" % (key, str(self.result[key])))
            writer.write('*' * 80)
            writer.write('\n')
        if eval_accuracy > self.best_acc:
            logger.info("=" * 80)
            logger.info("Best accuracy {}".format(eval_accuracy))
            logger.info("Saving Model......")
            self.best_acc = eval_accuracy
            self.best_coef = eval_coef
            # 保存best_coef
            np.save(os.path.join(output_path, "best_coef.npy"), self.best_coef)
            logger.info("Saving Best_coef......")
            # 保存模型
            model = self.model.module if hasattr(self.model, 'module') else self.model
            torch.save(model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
            logger.info("=" * 80)

    def test(self, test_loader):
        self.model.eval()
        inference_labels = []
        for input_ids, input_mask, segment_ids in test_loader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                test_output = self.model(input_ids=input_ids, token_type_ids=segment_ids,
                                         attention_mask=input_mask)
            # test_output = test_output[0]
            logits = test_output.detach().cpu().numpy()
            inference_labels.append(logits)

        logits = np.concatenate(inference_labels, 0).reshape(-1, self.args.num_classes)
        return logits

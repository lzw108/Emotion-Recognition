import gc
import logging
import os

import numpy as np
import pandas as pd
import torch
import hardvote
import random

import data_preprocess
from args import args
from dataset import load_data
from roberta_model import load_model
from running import Running

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_everything(seed=23):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train():
    # 训练epochs轮
    for epoch in range(args.epochs):
        # 载入模型
        tokenizer, model = load_model(args)
        # 读取数据
        data_path = args.data_path + str(epoch)
        train_data_path = os.path.join(data_path, 'train.csv')
        train_loader = load_data(tokenizer, args, train_data_path, "train")
        evaluate_data_path = os.path.join(data_path, 'dev.csv')
        evaluate_loader = load_data(tokenizer, args, evaluate_data_path, "evaluate")
        logger.info("Training data has been loaded!")
        # 训练
        running = Running(model, args)
        running.train(train_loader, evaluate_loader, epoch)
        # 释放显存
        torch.cuda.empty_cache()
        # 垃圾回收
        gc.collect()


def test():
    for epoch in range(args.epochs):
        # 载入模型和best_coef
        output_path = args.output_path + str(epoch)
        args.model_path = os.path.join(output_path, "pytorch_model.bin")
        tokenizer, model = load_model(args)
        best_coef = np.load(os.path.join(output_path, "best_coef.npy")).tolist()
        # 载入测试集数据
        data_path = args.data_path + str(epoch)
        test_data_path = os.path.join(data_path, 'test.csv')
        test_loader = load_data(tokenizer, args, test_data_path, "test")  # 3263
        logger.info("Testing data has been loaded!")
        # 得到测试结果
        running = Running(model, args)
        outputs = running.test(test_loader) * best_coef
        # 写入数据
        outputs_df = pd.read_csv(os.path.join(args.raw_data_path, "sample_submission.csv"))
        outputs_df['target_0'] = outputs[:, 0]
        outputs_df['target_1'] = outputs[:, 1]
        outputs_df['target_2'] = outputs[:, 2]
        outputs_df[['id', 'target_0', 'target_1', 'target_2']].to_csv(os.path.join(output_path, "sub.csv"), index=False)
        logger.info('sub ' + str(epoch) + ' has been written.')

#伪标记
def test_unlabel():
    for epoch in range(args.epochs):
        # 载入模型
        output_path = args.output_path + str(epoch)
        args.model_path = os.path.join(output_path, "pytorch_model.bin")
        tokenizer, model = load_model(args)
        # 载入测试集数据
        data_path = args.data_path + str(epoch)
        test_data_path = os.path.join(data_path, 'test.csv')
        test_loader = load_data(tokenizer, args, test_data_path, "test")  # 3263
        logger.info("Testing data has been loaded!")
        # 得到测试结果

        running = Running(model, args)
        outputs = running.test(test_loader)
        # 写入数据
        outputs_df = pd.read_csv(os.path.join(args.raw_data_path, "train.unlabeled.csv"))
        outputs_df['target_0'] = outputs[:, 0]
        outputs_df['target_1'] = outputs[:, 1]
        outputs_df['target_2'] = outputs[:, 2]
        outputs_df[['微博id','微博发布时间','发布人账号','微博中文内容','微博图片','微博视频', 'target_0', 'target_1', 'target_2']].to_csv(os.path.join(output_path, "sub.csv"), index=False)
        logger.info('sub ' + str(epoch) + ' has been written.')


# 生成最终结果-平均
def generate_result():
    submit_df = pd.read_csv(os.path.join(args.raw_data_path, "sample_submission.csv"))
    submit_df['0'] = 0
    submit_df['1'] = 0
    submit_df['2'] = 0
    for epoch in range(0, args.epochs):
        output_path = args.output_path + str(epoch)
        tmp = pd.read_csv(os.path.join(output_path, 'sub.csv'))
        submit_df['0'] += tmp['target_0'] / args.epochs
        submit_df['1'] += tmp['target_1'] / args.epochs
        submit_df['2'] += tmp['target_2'] / args.epochs
    submit_df['id'] = submit_df['id'].astype(str)
    submit_df['y'] = np.argmax(submit_df[['0', '1', '2']].values, -1) - 1
    submit_df[['id', 'y']].to_csv(os.path.join(args.submit_path, 'submit.csv'), index=False)
    logger.info("The final result has been generated.")

#伪标记
def generate_result_unlabel():
    submit_df = pd.read_csv(os.path.join(args.raw_data_path, "train.unlabeled.csv"))
    submit_df['0'] = 0
    submit_df['1'] = 0
    submit_df['2'] = 0
    for epoch in range(0, args.epochs):
        output_path = args.output_path + str(epoch)
        tmp = pd.read_csv(os.path.join(output_path, 'sub.csv'))
        submit_df['0'] += tmp['target_0'] / args.epochs
        submit_df['1'] += tmp['target_1'] / args.epochs
        submit_df['2'] += tmp['target_2'] / args.epochs
    submit_df['微博id'] = submit_df['微博id'].astype(str)
    submit_df['情感倾向'] = np.argmax(submit_df[['0', '1', '2']].values, -1) - 1
    submit_df[['微博id','微博发布时间','发布人账号','微博中文内容','微博图片','微博视频','情感倾向']].to_csv(os.path.join(args.submit_path, 'labeled.csv'), index=False)
    logger.info("The final result has been generated.")





if __name__ == '__main__':
    # 切分训练集和验证集数据
    seed_everything(seed=args.seed)
    data_preprocess.cut_fold(k=args.epochs)
    # 训练
    train()

    if args.Pseudo:
        test_unlabel()
        generate_result_unlabel()
    else:
        test()
        generate_result()
        hardvote.generate_hard()


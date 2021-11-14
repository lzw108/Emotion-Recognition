import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel, BertTokenizer, BertConfig


class BertForClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForClassification, self).__init__(config)
        config.output_hidden_states = True  # 是否输出hidden states

        self.num_labels = 2  # config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.AlphaDropout(p=0.2)
        self.high_dropout = nn.AlphaDropout(p=0.35)

        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        outputs = self.bert(input_ids=flat_input_ids, position_ids=flat_position_ids,
                            token_type_ids=flat_token_type_ids,
                            attention_mask=flat_attention_mask, head_mask=head_mask)

        hidden_layers = outputs[2]

        stacks = []
        for i in range(len(hidden_layers)):
            if i != len(hidden_layers) - 1:
                stacks.append(self.dropout(hidden_layers[i][:, 0, :]))
            else:
                stacks.append(hidden_layers[i][:, 0, :])
        cls_outputs = torch.stack(stacks, dim=2)


        # batch_size, hidden_size, hidden_layers + 1

        #cls_outputs = torch.stack([self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2)

        # batch_size, hidden_size
        cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)



        # multisample dropout (wut): https://arxiv.org/abs/1905.09788
        # batch_size, num_classes
        logits = torch.mean(torch.stack([self.classifier(self.high_dropout(cls_output)) for _ in range(8)],
                                        dim=0), dim=0)



        if labels is not None:
            # loss_fct = BCEWithLogitsLoss(reduction='mean')
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # labels = torch.zeros(logits.shape, device=device).scatter_(1, torch.unsqueeze(labels, dim=1), 1)
            loss_fct = CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


def load_model(args):
    # 载入模型
    # Bert
    tokenizer = BertTokenizer.from_pretrained(args.config_path, do_lower_case=args.do_lower_case)
    config = BertConfig.from_pretrained(args.config_path, num_labels=args.num_classes)
    model = BertForClassification.from_pretrained(args.model_path, config=config)
    return tokenizer, model




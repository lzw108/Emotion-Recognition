import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler


def load_data(tokenizer, args, data_path, data_type):
    df = pd.read_csv(data_path)
    examples = []
    # 提取出第三列作为训练特征
    for text in df[list(df.columns[3:4])].values:
        examples.append(text[0])

    input_ids, input_masks, input_segments = convert_examples_to_features(tokenizer, args, examples)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.long)
    input_segments = torch.tensor(input_segments, dtype=torch.long)

    if data_type is "test":
        data_set = TensorDataset(input_ids, input_masks, input_segments)
        data_sampler = SequentialSampler(data_set)
        data_loader = DataLoader(data_set, sampler=data_sampler, batch_size=args.eval_batch_size)
    else:
        labels = list(df["情感倾向"])
        labels = torch.tensor(labels, dtype=torch.long) + 1
        data_set = TensorDataset(input_ids, input_masks, input_segments, labels)
        if data_type is "train":
            data_sampler = RandomSampler(data_set)
            data_loader = DataLoader(data_set, sampler=data_sampler, batch_size=args.batch_size)
        else:
            data_sampler = SequentialSampler(data_set)
            data_loader = DataLoader(data_set, sampler=data_sampler, batch_size=args.eval_batch_size)

    return data_loader


def convert_examples_to_features(tokenizer, args, examples):
    input_ids, input_masks, input_segments = [], [], []
    for example in examples:
        text_token = tokenizer.tokenize(example)

        text_length = args.max_sequence_length - 2
        text_token = text_token[:text_length]
        # tokens的组成
        # tokens = ["[CLS]"] + keyword_token + ["[SEP]"] + location_token + ["[SEP]"] + text_token + ["[SEP]"]
        tokens = ["[CLS]"] + text_token + ["[SEP]"]

        input_id = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_id)
        # input_segment = [0] * (len(keyword_token) + 2) + [0] * (len(location_token) + 1) + [1] * (
        #             len(text_token) + 1)
        input_segment = [1] * (len(text_token) + 2)
        padding_length = args.max_sequence_length - len(input_id)
        input_id += ([0] * padding_length)
        input_mask += ([0] * padding_length)
        input_segment += ([0] * padding_length)

        input_ids.append(input_id)
        input_masks.append(input_mask)
        input_segments.append(input_segment)
    return input_ids, input_masks, input_segments


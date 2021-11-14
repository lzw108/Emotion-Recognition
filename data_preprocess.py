# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from args import args
import logging
from clean_data import process_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_text_length(filename):
    # 文本长度的范围
    df = pd.read_csv(filename)
    texts = df['微博中文内容']
    texts.fillna("无", inplace=True)

    # 获取所有文本的长度
    all_length = []
    for content in texts:
        try:
            all_length.append(len(content))
        except Exception:
            print(content)

    plt.hist(all_length, bins=30)
    plt.show()
    print(np.mean(np.array(all_length) < 170))
    df[:10].to_csv("./data/train.csv", index=False)


# 可视化语料序列长度, 可见文本的长度都在160以下
# plot_text_length('./data/train0.csv')
# plot_text_length('./data/test0.csv')
random.seed(23)

def cut_fold(k):
    train_df = pd.read_csv("./data/train.csv",encoding="utf-8")
    test_df = pd.read_csv("./data/test.csv",encoding="utf-8")
    #给未标注数据集合标注
    unlabeled_df = pd.read_csv("./data/train.unlabeled.csv", encoding="utf-8")
    #伪标签
    train_pseudo_df = pd.read_csv("./data/labeled.csv", encoding="utf-8")

    train_df['微博中文内容'].fillna('无', inplace=True)
    test_df['微博中文内容'].fillna('无', inplace=True)
    unlabeled_df['微博中文内容'].fillna('无', inplace=True)
    train_pseudo_df['微博中文内容'].fillna('无', inplace=True)

    train_df = train_df.loc[train_df["情感倾向"].isin(['-1', '0', '1'])]
    #pseudo_df = pseudo_df.loc[train_df["情感倾向"].isin(['-1', '0', '1'])]

    # 数据清洗
    train_df_unpseudo = process_text(train_df)
    test_df = process_text(test_df)
    unlabeled_df = process_text(unlabeled_df)
    train_pseudo_df = process_text(train_pseudo_df)

    #数据融合
    if args.Pseudo_train:
        train_pseudo_df = train_pseudo_df.sample(frac=0.9)
        train_df = pd.concat([train_df_unpseudo, train_pseudo_df], axis=0)
        logger.info("Pseudo is train")
    else:
        train_df = train_df_unpseudo

    #伪标记
    if args.Pseudo:
        test_df = unlabeled_df
        logger.info("Pseudo")

    index_unpseudo = set(range(train_df_unpseudo.shape[0]))
    index = set(range(train_df.shape[0]))
    train_fold = []
    dev_fold = []
    for i in range(k):
        dev = random.sample(index_unpseudo,11000)
        train = index - set(dev)
        print("Dev Number:", len(dev))
        print("Train Number:", len(train))
        dev_fold.append(dev)
        train_fold.append(train)

    for i in range(k):
        print("Fold", i)
        path = "./data/data_" + str(i)
        if not os.path.exists(path):
            os.makedirs(path)
        dev_index = list(dev_fold[i])
        train_index = list(train_fold[i])
        train_df.iloc[train_index].to_csv("./data/data_{}/train.csv".format(i), index=False)
        train_df.iloc[dev_index].to_csv("./data/data_{}/dev.csv".format(i), index=False)
        test_df.to_csv("./data/data_{}/test.csv".format(i), index=False)


#cut_fold(5)

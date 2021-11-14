import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from args import args

def work(pres):
    count = [0, 0, 0]
    for i in pres:
        count[i] += 1
    out = count.index(max(count))
    return out


def generate_hard():

    submit_df = pd.read_csv(os.path.join(args.raw_data_path, "sample_submission.csv"))
    for epoch in range(0, args.epochs):
        output_path = args.output_path + str(epoch)
        tmp = pd.read_csv(os.path.join(output_path, 'sub.csv'))
        submit_df['epoch' + str(epoch)] = np.argmax(tmp[['target_0', 'target_1', 'target_2']].values, axis=1) - 1

    submit_df.drop('y', axis=1, inplace=True)
    submit_df.to_csv(os.path.join(args.submit_path,'subs.csv'), index=False)
    tmp = np.array(submit_df.iloc[:, 1:]) + 1
    label_voted = [work(line) for line in tmp]
    submit_df['y'] = label_voted
    submit_df['y'] = submit_df.y - 1
    submit_df[['id', 'y']].to_csv(os.path.join(args.submit_path, 'hard_submit.csv'), index=False)
    logger.info("The final result has been generated use hardvote.")




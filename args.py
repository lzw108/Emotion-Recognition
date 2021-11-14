import argparse

parser = argparse.ArgumentParser()

# 各个文件路径
parser.add_argument("--raw_data_path", type=str, default="./data")
parser.add_argument("--data_path", type=str, default="./data/data_")
parser.add_argument("--submit_path", type=str, default="./output")
parser.add_argument("--model_path", type=str, default="./roberta_model")
parser.add_argument("--output_path", type=str, default="./output/data_")
parser.add_argument("--config_path", type=str, default="./roberta_model")
parser.add_argument("--seed", type=int, default=42)

# 超参数
parser.add_argument("--train_steps", type=int, default=4000)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--eval_batch_size", type=int, default=64)
parser.add_argument("--batch_accumulation", type=int, default=4)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=5e-6)
parser.add_argument("--warmup", type=int, default=100)

# loss
parser.add_argument("--num_classes", type=int, default=3)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--eval_steps", type=int, default=200)

# swa
parser.add_argument('--swa', type=bool, default=False, help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=200, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR', help='SWA LR (default: 0.05)')
parser.add_argument('--swa_freq', type=int, default=100, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

# tokenization
parser.add_argument("--do_lower_case", type=bool, default=False, help="Set this flag if you are using an uncased model.")
parser.add_argument("--max_sequence_length", type=int, default=170)

#伪标记
parser.add_argument('--Pseudo', type=bool, default=False, help='Pseudo usage flag (default: off)')
parser.add_argument('--Pseudo_train', type=bool, default=True, help='Pseudo_train usage flag (default: off)')

#




args = parser.parse_args()

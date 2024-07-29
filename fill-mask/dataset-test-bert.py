from transformers import pipeline, BertTokenizer
import transformers
import deepspeed
import torch
import os
import argparse
import math
import time
from deepspeed.accelerator import get_accelerator
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, help="hf model name")
parser.add_argument("--dtype", type=str, default="fp16", help="fp16 or fp32")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
parser.add_argument("--trials", type=int, default=8, help="number of trials")
parser.add_argument("--kernel-inject", action="store_true", help="inject kernels on")
parser.add_argument("--graphs", action="store_true", help="CUDA Graphs on")
parser.add_argument("--triton", action="store_true", help="triton kernels on")
parser.add_argument("--deepspeed", action="store_true", help="use deepspeed inference")
parser.add_argument("--task", type=str, default="fill-mask", help="fill-mask or token-classification")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--max_new_tokens", default=50, type=int, help="maximum new tokens to generate")
args = parser.parse_args()

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '2'))

os.environ['NCCL_SOCKET_IFNAME']="eno1"
# print(f"local_rank: {local_rank}")
# print(f"world_size: {world_size}")

pipe = pipeline('fill-mask', model='./bert-base', device=local_rank)

# pipe = pipeline('fill-mask', model='./bert-base', device=local_rank, batch_size=args.batch_size)
# dataset = load_dataset("parquet", data_files={'./wikipedia/data/20220301.en/train-00000-of-00041.parquet'})
dataset = load_dataset("./wikipedia", "20220301.en", "train-00000-of-00041.parquet")
print(dataset["train"][0]["text"])

pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float16 if args.triton else torch.float
)

pipe.device = torch.device(f'cuda:{local_rank}')

# for out in pipe(dataset["train"][0]["text"], batch_size=args.batch_size):
#     print(out)

# for out in pipe(KeyDataset(dataset, "text"))
#     print(out)
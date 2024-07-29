from transformers import pipeline
import transformers
import deepspeed
import torch
import argparse
import os
import math
import time
from transformers.models.t5.modeling_t5 import T5Block

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
parser.add_argument("--hostfile", type=str)
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '2'))

os.environ['NCCL_SOCKET_IFNAME']="eno1"

# tokenizer_name = "t5-v1_tokenizer"
model_name = "t5-v1"

# pipe = pipeline("text2text-generation", model = model_name, tokenizer = tokenizer_name, device=local_rank)
pipe = pipeline("text2text-generation", model = model_name, device=local_rank, batch_size = args.batch_size)

# The injection_policy shows two things:
#   1. which layer module we need to add Tensor-Parallelism
#   2. the name of several linear layers: a) attention_output (both encoder and decoder), 
#       and b) transformer output

pipe.model = deepspeed.init_inference( 
    pipe.model,
    mp_size=world_size,
    dtype=torch.float,
    tp_proportion=(4,2)
)
pipe.model.profile_model_time(use_cuda_events=True)
pipe.device = torch.device(f'cuda:{local_rank}')

input_sentences = [
    "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy"
]
if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

inputs = input_sentences[:args.batch_size]

e2e_times = []
model_times = []

iters = 30
for i in range(iters):
    # get_accelerator().synchronize()
    start = time.time()

    outputs = pipe(inputs)

    # get_accelerator().synchronize()
    end = time.time()
    
    e2e_times.append((end - start) * 1e3)  # convert ns to ms
    model_times.extend(pipe.model.model_times())

# output = pipe("Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy")

rank = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()

if rank == 0 or rank == 2:
    # print(f"model_times: {len(model_times)}")
    # print(outputs)
    warmup = 3
    e2e_times = e2e_times[warmup:]
    model_times = model_times[warmup:]
    print(f"rank: {rank} end2end time is {sum(e2e_times)/(len(e2e_times))} ms")
    print(f"rank: {rank} model time is {sum(model_times)/(len(e2e_times))} ms")
    # print(outputs)
from transformers import pipeline, AutoTokenizer
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
parser.add_argument("--hostfile", type=str)
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '2'))

# print(args.hostfile)

os.environ['NCCL_SOCKET_IFNAME']="eno1"
# print(f"local_rank: {local_rank}")
# print(f"world_size: {world_size}")

model_name = './roberta-large'


pipe = pipeline('fill-mask', model=model_name, device=local_rank, batch_size = args.batch_size)

# print("create pipeline")

pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float32,
    tp_proportion=(7,1)
)
pipe.model.profile_model_time(use_cuda_events=True)
pipe.device = torch.device(f'cuda:{local_rank}')

# print("create inference engine")

# input_sentences = [
#     "In Autumn the <mask> fall from the trees.",
#     "He has a big <mask>.",
#     "Paris is the <mask> of France.",
#     "The goal of life is <mask>."
# ]
input_sentences = [
    "zero-shot-classification and question-answering are slightly <mask> in the sense, \
        that a single input might yield multiple forward pass of a model.",
    "And remember, it's not the <mask> that stand by your side when you're at your best, \
        but the ones who stand beside you when you're at your worst that are your true friends.",
    "In the flood of darkness, <mask> is the light. It brings comfort, faith, and confidence. It gives us guidance when we are lost, \
        and gives support when we are afraid. ",
    "Only when you understand the true meaning of life can you live truly. Bittersweet as life is, it's still wonderful, \
        and it's <mask> even in tragedy.",
    "I believe there is a person who brings sunshine into your life. \
        But if you really have to wait for someone to bring you the <mask> and give you a good feeling, then you may have to wait a long time."
]
if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

inputs = input_sentences[:args.batch_size]


prof = torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=3, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_log/'+ model_name[2:] +'/ds_'+str(args.batch_size)),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
)
prof.start()
# times = []
e2e_times = []
model_times = []

print("start inference")

iters = 30
for i in range(iters):
    # get_accelerator().synchronize()
    start = time.time()

    outputs = pipe(inputs)

    # get_accelerator().synchronize()
    end = time.time()
    
    e2e_times.append((end - start) * 1e3)  # convert ns to ms
    model_times.extend(pipe.model.model_times())
    prof.step()

prof.stop()



rank = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()

if rank == 0 or rank == 2:
    # print(f"model_times: {len(model_times)}")
    warmup = 3
    e2e_times = e2e_times[warmup:]
    model_times = model_times[warmup:]
    print(f"rank: {rank} end2end time is {sum(e2e_times)/(len(e2e_times))} ms")
    print(f"rank: {rank} model time is {sum(model_times)/(len(e2e_times))} ms")
    # print(outputs)
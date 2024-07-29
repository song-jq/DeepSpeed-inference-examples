from transformers import pipeline, AutoTokenizer
import transformers
import deepspeed
import torch
import torch.cuda as cuda
import os
import argparse
import math
import time
from deepspeed.accelerator import get_accelerator
from datasets import load_dataset
from transformers.models.bert.modeling_bert import BertLayer, BertConfig, BertEncoder, BertModel
import getpass

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
username = getpass.getuser()
model_name = '/home/' + username + '/Data/huggingface/models' + '/bert-base'

# print("start")

# times1 = 2
# times2 = 1
# config = BertConfig()
# config = BertConfig(hidden_size = 1024 * times1, intermediate_size = 4096 * times1, num_attention_heads = 16 * times2, num_hidden_layers = 16 * times2)
# print("create bert config")
# bert = BertModel(config)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# print("create bert")
# pipe = pipeline('fill-mask', model=bert, tokenizer=tokenizer, device=local_rank)


pipe = pipeline('fill-mask', model=model_name, device=local_rank, batch_size = args.batch_size)

# print("create pipeline")

pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float32
)
pipe.model.profile_model_time(use_cuda_events=True)

pipe.device = torch.device(f'cuda:{local_rank}')

# print("create inference engine")

# input_sentences = [
#     "In Autumn the [MASK] fall from the trees.",
#     "He has a big [MASK].",
#     "Paris is the [MASK] of France.",
#     "The goal of life is [MASK]."
# ]
input_sentences = [
    "zero-shot-classification and question-answering are slightly [MASK] in the sense, \
        that a single input might yield multiple forward pass of a model.",
    "And remember, it's not the [MASK] that stand by your side when you're at your best, \
        but the ones who stand beside you when you're at your worst that are your true friends.",
    "In the flood of darkness, [MASK] is the light. It brings comfort, faith, and confidence. It gives us guidance when we are lost, \
        and gives support when we are afraid. ",
    "Only when you understand the true meaning of life can you live truly. Bittersweet as life is, it's still wonderful, \
        and it's [MASK] even in tragedy.",
    "I believe there is a person who brings sunshine into your life. \
        But if you really have to wait for someone to bring you the [MASK] and give you a good feeling, then you may have to wait a long time."
]
if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

inputs = input_sentences[:args.batch_size]


# prof = torch.profiler.profile(
#     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#     schedule=torch.profiler.schedule(wait=1, warmup=3, active=3, repeat=1),
#     on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_log/'+ model_name[2:] +'/ds_'+str(args.batch_size)),
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=True
# )
# prof.start()
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
#     prof.step()

# prof.stop()


rank = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()

if rank == 0 or rank == 2:
    # print(f"model_times: {len(model_times)}")
    warmup = 3
    e2e_times = e2e_times[warmup:]
    model_times = model_times[warmup:]
    print(f"rank: {rank} end2end time is {sum(e2e_times)/(len(e2e_times))} ms")
    print(f"rank: {rank} model time is {sum(model_times)/(len(e2e_times))} ms")
    # print(outputs)


# gpu_count = cuda.device_count()
# print("可用的GPU数量:", gpu_count)

# free_memory_tuple_list = []
# for i in range(gpu_count):
#     props = cuda.get_device_properties(i)
#     total_memory = props.total_memory // 1024**2  # 转换为MB
#     allocated_memory = cuda.memory_allocated(i) // 1024**2  # 转换为MB
#     free_memory = total_memory - allocated_memory
#     print("GPU {} 总内存: {}MB".format(i, total_memory))
#     print("GPU {} 可用内存: {}MB".format(i, free_memory))
#     memory_tuple=(rank, free_memory)
#     free_memory_tuple_list.append(memory_tuple)

# print(free_memory_tuple_list)

# all_memory_tuple_list = []

# torch.distributed.all_gather(all_memory_tuple_list, free_memory_tuple_list)
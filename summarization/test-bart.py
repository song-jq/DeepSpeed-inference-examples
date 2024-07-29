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
from transformers.models.bert.modeling_bert import BertLayer, BertConfig, BertEncoder, BertModel

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

model_name = './bart-base'


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


pipe = pipeline('summarization', model=model_name, device=local_rank, batch_size = args.batch_size)

# print("create pipeline")

pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float32,
    tp_proportion=(5,1)
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
    """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""
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

iters = 30
for i in range(iters):
    # get_accelerator().synchronize()
    start = time.time()

    outputs = pipe(inputs, max_length=130, min_length=30, do_sample=False)
    # get_accelerator().synchronize()
    end = time.time()
    
    e2e_times.append((end - start) * 1e3)  # convert ns to ms
    model_times.extend(pipe.model.model_times())
#     prof.step()

# prof.stop()



rank = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()

if rank == 0 or rank == 2:
    # print(outputs)
    warmup = 3
    e2e_times = e2e_times[warmup:]
    model_times = model_times[warmup:]
    print(f"rank: {rank} end2end time is {sum(e2e_times)/(len(e2e_times))} ms")
    print(f"rank: {rank} model time is {sum(model_times)/(len(e2e_times))} ms")
    # print(outputs)
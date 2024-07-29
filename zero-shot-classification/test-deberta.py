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

model_name = './debarta-large'


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


pipe = pipeline('zero-shot-classification', model=model_name, device=local_rank, batch_size = args.batch_size)

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
#     "In Autumn the [MASK] fall from the trees.",
#     "He has a big [MASK].",
#     "Paris is the [MASK] of France.",
#     "The goal of life is [MASK]."
# ]
input_sentences = [""" A new model offers an explanation for how the Galilean satellites formed around the solar system’s largest world. Konstantin Batygin did not set out to solve one of the solar system’s most puzzling mysteries when he went for a run up a hill in Nice, France. Dr. Batygin, a Caltech researcher, best known for his contributions to the search for the solar system’s missing “Planet Nine,” spotted a beer bottle. At a steep, 20 degree grade, he wondered why it wasn’t rolling down the hill. He realized there was a breeze at his back holding the bottle in place. Then he had a thought that would only pop into the mind of a theoretical astrophysicist: “Oh! This is how Europa formed.” Europa is one of Jupiter’s four large Galilean moons. And in a paper published Monday in the Astrophysical Journal, Dr. Batygin and a co-author, Alessandro Morbidelli, a planetary scientist at the Côte d’Azur Observatory in France, present a theory explaining how some moons form around gas giants like Jupiter and Saturn, suggesting that millimeter-sized grains of hail produced during the solar system’s formation became trapped around these massive worlds, taking shape one at a time into the potentially habitable moons we know today."""]
candidate_labels = ["space & cosmos", "scientific discovery", "microbiology", "robots", "archeology"]

# input_sentences = ["Angela Merkel is a politician in Germany and leader of the CDU"]
# candidate_labels = ["politics", "economy", "entertainment", "environment"]

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

    outputs = pipe(inputs, candidate_labels, multi_label=False)
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
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BertForMaskedLM, BertTokenizer
import transformers
import deepspeed
import torch
import os
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertLayer, BertConfig, BertEncoder
from deepspeed.accelerator import get_accelerator
import time

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '2'))
os.environ['NCCL_SOCKET_IFNAME']="eno1"

model = BertForMaskedLM.from_pretrained("./bert-base")
tokenizer = BertTokenizer.from_pretrained("./bert-base")
device = torch.device(f'cuda:{local_rank}')

# input_ids = tokenizer("I have a [MASK].",return_tensors = "pt")
# input_ids = input_ids.to(device)
# print(input_ids)

times = 2
hidden_size = 1024 * times

batch_size = 8
sequence_length = 1024

config = BertConfig(hidden_size=hidden_size, intermediate_size=hidden_size*4, num_attention_heads=16, num_hidden_layers=16)
# print(config)
encoder = BertEncoder(config)
print(encoder)

hidden_states = torch.HalfTensor(batch_size, sequence_length, hidden_size)
hidden_states = hidden_states.to(device)

engine = deepspeed.init_inference(
    encoder,
    mp_size=world_size,
    dtype=torch.float16,
    injection_policy={BertLayer: ('output.dense')}
)
engine.profile_model_time(use_cuda_events=True)
# pipemodel = engine.module


# prof = torch.profiler.profile(
#     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#     schedule=torch.profiler.schedule(wait=1, warmup=3, active=3, repeat=1),
#     on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_log/'+ 'test_layer' +'/times_' + str(times) 
#         + '-sequence_length_' + str(sequence_length) + '-batch_size_' + str(batch_size)),
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=True
# )
# prof.start()

e2e_times = []
model_times = []

iters = 30
for i in range(iters):
    get_accelerator().synchronize()
    start = time.time()

    output = engine(hidden_states)
    get_accelerator().synchronize()
    end = time.time()
    
    e2e_times.append((end - start) * 1e3)  # convert ns to ms
    model_times.extend(engine.model_times())
#     prof.step()

# prof.stop()

rank = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()

# if rank == 0 or rank == 2:
    # print(f"model_times: {len(model_times)}")
warmup = 3
e2e_times = e2e_times[warmup:]
model_times = model_times[warmup:]
print(f"rank: {rank} end2end time is {sum(e2e_times)/(len(e2e_times))} ms")
print(f"rank: {rank}  model  time is {sum(model_times)/(len(model_times))} ms")
    # print(outputs)




# attention_mask = torch.full([20, 55], -65504.)
# attention_mask = attention_mask.to(device)
# head_mask = [None, None, None, None, None, None, None, None, None, None, None, None]
# encoder_hidden_states = None
# encoder_attention_mask = None
# past_key_values = None
# use_cache = False
# output_attentions = False
# output_hidden_states = False
# return_dict = True

# attention_mask = torch.full([20, 55], -65504., dtype=torch.float16) #[20,1,1,55]
# attention_mask = torch.triu(attention_mask, 36)
# attention_mask = torch.unsqueeze(attention_mask,1)
# attention_mask = torch.unsqueeze(attention_mask,1)
# attention_mask = attention_mask.to(device)

# input = {'hidden_states':hidden_states, 'attention_mask':attention_mask, 'head_mask':head_mask, 'encoder_hidden_states':encoder_hidden_states, 
#     'encoder_attention_mask':encoder_attention_mask, 'past_key_values':past_key_values, 'use_cache':use_cache, 'output_attentions':output_attentions, 
#         'output_hidden_states':output_hidden_states, 'return_dict':return_dict}

# config = {"_name_or_path": "./bert-base",
#   "architectures": [
#     "BertForMaskedLM"
#   ],
#   "attention_probs_dropout_prob": 0.1,
#   "classifier_dropout": None,
#   "gradient_checkpointing": False,
#   "hidden_act": "gelu",
#   "hidden_dropout_prob": 0.1,
#   "hidden_size": 768,
#   "initializer_range": 0.02,
#   "intermediate_size": 3072,
#   "layer_norm_eps": 1e-12,
#   "max_position_embeddings": 512,
#   "model_type": "bert",
#   "num_attention_heads": 12,
#   "num_hidden_layers": 12,
#   "pad_token_id": 0,
#   "position_embedding_type": "absolute",
#   "transformers_version": "4.28.1",
#   "type_vocab_size": 2,
#   "use_cache": True,
#   "vocab_size": 30522}
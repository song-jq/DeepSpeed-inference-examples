#!/bin/bash
scp test-bart.py kechengsheji@10.0.16.13:~/inference/summarization
scp -r /home/kechengsheji/DeepSpeed/deepspeed/module_inject kechengsheji@10.0.16.13:/home/kechengsheji/DeepSpeed/deepspeed

batch_size=128

deepspeed --num_gpus 2 --hostfile hostfile.txt test-bart.py --batch_size $batch_size > output.txt


# deepspeed --num_gpus 2 --hostfile hostfile.txt test-bert.py --batch_size 512 > output.txt
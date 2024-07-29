#!/bin/bash
scp test-t5.py kechengsheji@10.0.16.13:~/inference/translation
scp -r /home/kechengsheji/DeepSpeed/deepspeed/module_inject kechengsheji@10.0.16.13:/home/kechengsheji/DeepSpeed/deepspeed

batch_size=16

deepspeed --num_gpus 2 --hostfile hostfile.txt test-t5.py --batch_size $batch_size > output.txt

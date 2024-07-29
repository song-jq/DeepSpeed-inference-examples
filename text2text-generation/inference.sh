#!/bin/bash
scp test-t5.py kechengsheji@10.0.16.13:~/inference/text2text-generation
# scp test-bert.py kechengsheji@10.0.16.13:~/inference/fill-mask
# scp test-roberta.py kechengsheji@10.0.16.13:~/inference/fill-mask
scp -r /home/kechengsheji/DeepSpeed/deepspeed/module_inject kechengsheji@10.0.16.13:/home/kechengsheji/DeepSpeed/deepspeed

batch_size=256

deepspeed --num_gpus 2 --hostfile hostfile.txt test-t5.py --batch_size $batch_size > output.txt

# if [ "$1" == "bert" ]; then
#     echo "call test-bert"
#     deepspeed --num_gpus 2 --hostfile hostfile.txt test-bert.py --batch_size $batch_size > output.txt
# elif [ "$1" == "roberta" ];then
#     echo "call test-roberta"
#     deepspeed --num_gpus 2 --hostfile hostfile.txt test-roberta.py --batch_size $batch_size > output.txt
# else
#     echo "call test-layer"
#     deepspeed --num_gpus 2 --hostfile hostfile.txt test-layer.py > output.txt
# fi


# deepspeed --num_gpus 2 --hostfile hostfile.txt test-bert.py --batch_size 512 > output.txt
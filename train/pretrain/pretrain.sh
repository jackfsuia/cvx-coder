#!/bin/bash
# export CUDA_DEVICE_MAX_CONNECTIONS=1

# # ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# # See the section for finetuning in README for more information.
DEBUGPY=False
function usage(){
    echo 'Usage: bash finetune/finetune_ds.sh [-m MODEL_PATH] [-d DATA_PATH]'
}
while [[ "$1" != "" ]]; do
    case $1 in
        -b | --debug )
            shift
            DEBUGPY=True
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done
# # Number of GPUs per GPU worker
# GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# # Number of GPU workers, for single-worker training, please set to 1
# NNODES=${NNODES:-1}

# # The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
# NODE_RANK=${NODE_RANK:-0}

# # The ip address of the rank-0 worker, for single-worker training, please set to localhost
# MASTER_ADDR=${MASTER_ADDR:-localhost}

# # The port for communication
# MASTER_PORT=${MASTER_PORT:-6001}

# DISTRIBUTED_ARGS="
#     --nproc_per_node $GPUS_PER_NODE \
#     --nnodes $NNODES \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT
# "
# echo "The value of my_variable is: $GPUS_PER_NODE and $NNODES"
# torchrun $DISTRIBUTED_ARGS finetune_new.py \
python pretrain.py \
    --model_name_or_path "/data/Phi-3-mini-4k-instruct" \
    --data_path "/data/Phi-3-mini-4k-instruct/CVX.md"\
    --eval_data_path "/data/Phi-3-mini-4k-instruct/test_merge copy.jsonl" \
    --output_dir "/data/Phi-3-mini-4k-instruct/outputmodels2" \
    --num_train_epochs 20 \
    --bf16 True \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 5 \
    --gradient_accumulation_steps 12 \
    --eval_strategy "steps" \
    --eval_steps 1\
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 4 \
    --logging_steps 1 \
    --learning_rate 5e-4 \
    --warmup_ratio 0 \
    --lr_scheduler_type "cosine" \
    --log_level 'info'\
    --logging_dir "/data/Phi-3-mini-4k-instruct/outputmodels2/logs" \
    --logging_strategy "steps" \
    --report_to "tensorboard" \
    --model_max_length 1500 \
    --gradient_checkpointing True \
    --debugpy $DEBUGPY \
    --evaluate_before_train True \
    --lora_target_modules 'down_proj' 'gate_up_proj' 'o_proj' 'qkv_proj' 'gate_proj' 'up_proj' 'down_proj' \
    --modules_to_save 'embed_tokens'  'lm_head'\
    --use_lora True\
    --lora_r 30 \
    --overwrite_output_dir True

    # --model_checkpoint_bin "/data/Qwen1.5-0.5B-Chat/outputmodels-b/checkpoint-36/model.safetensors" \

    # --fp16 False \
#   --model_checkpoint_bin "/data/chunk_slm/outputModels2/checkpoint-7500/model.safetensors" \
# --bf16 False \
# --resume_from_checkpoint "/data/chunk_slm/outputModels/checkpoint-15000" \
#   --use_lora \
#   --modules_to_save 'embed_tokens'  'lm_head'\
#   --lora_target_modules 'q_proj' 'k_proj' 'v_proj' 'o_proj' 'gate_proj' 'up_proj' 'down_proj' \
#   --deepspeed ${DS_CONFIG_PATH}
  #   --logging_steps 10 \
#   --fp16 \
#   --deepspeed ${DS_CONFIG_PATH}
#   --use_cpu True
#   --lora_target_modules "["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj","up_proj","down_proj"]" \
#   --q_lora False
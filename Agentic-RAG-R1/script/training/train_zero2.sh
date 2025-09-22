source ~/venv/bin/activate
set -a; source .env; set +a

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1

accelerate launch \
  --config_file ./src/config/accelerate_config/train_zero2.yaml \
  --num_processes 2 \
  --main_process_port 12347 \
  --mixed_precision fp16 \
  ./src/train.py \
    --config ./src/config/train_small.json \
    --use_lora true \
    --lora_r 32 \
    --load_in_4bit true \
    --gradient_checkpointing true \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1

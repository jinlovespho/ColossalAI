set -xe
# pip install -r requirements.txt

# number of gpus to use
GPUNUM=2

DISTRIBUTED_ARGS="
  --standalone \
  --nproc_per_node ${GPUNUM} \
"

TRAINING_ARGS="
  --num_epoch 3 \
  --batch_size 128 \
"
# plugin(training strategy)
# can only be one of "torch_ddp"/"torch_ddp_fp16"/"low_level_zero"/"gemini"/"hybrid_parallel"
PLUGIN="hybrid_parallel"

PARALLEL_ARGS="
  --plugin ${PLUGIN} \
  --tp_size 1 \
  --pp_size 1 \
"

MODEL_ARGS="
  --model_name vit_tiny_8 \
  --output_path ./output_model \
  --learning_rate 1e-3 \
  --dropout_ratio 0.1 \
  --warmup_ratio 0.3 \
  --weight_decay 0.1 \
  --seed 42 \
  --model_name_or_path google/vit-base-patch16-224 \
"

DATA_ARGS="
  --dataset cifar100 \
  --num_class 100 \
  --img_size 32 \
  --patch_size 8 \
"

CUDA_VISIBLE_DEVICES=0,2 torchrun ${DISTRIBUTED_ARGS} vit_train_demo.py ${PARALLEL_ARGS} ${MODEL_ARGS} ${DATA_ARGS}


# run the script for demo
# colossalai run \
#   --nproc_per_node ${GPUNUM} \
#   --master_port 29505 \
#   vit_train_demo.py \
#   --model_name_or_path ${MODEL} \
#   --output_path ${OUTPUT_PATH} \
#   --plugin ${PLUGIN} \
#   --batch_size ${BS} \
#   --tp_size ${TP_SIZE} \
#   --pp_size ${PP_SIZE} \
#   --num_epoch ${EPOCH} \
#   --learning_rate ${LR} \
#   --weight_decay ${WEIGHT_DECAY} \
#   --warmup_ratio ${WARMUP_RATIO}




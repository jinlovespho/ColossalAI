set -xe
# pip install -r requirements.txt

dataset="cifar100"
DATA_ARGS="
  --data_dir /media/dataset1/lig_jinlovespho/cifar100 \
  --dataset ${dataset} \
  --num_class 100 \
  --img_size 32 \
"

PLUGIN="hybrid_parallel"  # plugin(training strategy): "torch_ddp"/"torch_ddp_fp16"/"low_level_zero"/"gemini"/"hybrid_parallel"
tp_size=2
PARALLEL_ARGS="
  --plugin ${PLUGIN} \
  --tp_size ${tp_size} \
  --pp_size 1 \
"

lr=1e-3
lr_scheduler="linear"   # 'linear', 'cosine'
TRAINING_ARGS="
  --num_epoch 200 \
  --batch_size 128 \
  --lr_scheduler ${lr_scheduler} \
  --learning_rate ${lr} \
  --weight_decay 0.0001 \
  --label_smoothing \
  --seed 42 \
"

model_name="vit_small"   # 'vit_tiny' 'vit_small' 'vit_base' 'vit_large' 'vit_splithead_tiny/small/base'
patch_size=4
splithead_method=0    # 0:linear, 1:featurewise, 2:featureconv, 3:shuffle 4:roll
MODEL_ARGS="
  --model_name ${model_name} \
  --patch_size ${patch_size} \
  --output_path ./output_model \
  --dropout_ratio 0.1 \
  --warmup_ratio 0.01 \
  --splithead_method ${splithead_method} \
"


is_wandb="disabled"   # ['disabled', 'online']
WANDB_ARGS="
--is_wandb ${is_wandb} \
--project_name lignex1_vit_cifar100 \
--exp_name gpu23_tp${tp_size}_${dataset}_${lr_scheduler}lr${lr}_${model_name}${patch_size} \
--wandb_save_dir /media/dataset1/lig_jinlovespho/log/colAI \
"

# number of gpus to use
GPUNUM=2

DISTRIBUTED_ARGS="
  --standalone \
  --nproc_per_node ${GPUNUM} \
"

# 아래 명령으로 디버깅 가능
# NCCL_DEBUG=INFO bash my_vit_cifar100_tp2.sh
CUDA_VISIBLE_DEVICES=2,3 torchrun ${DISTRIBUTED_ARGS} my_vit_train.py ${TRAINING_ARGS} ${PARALLEL_ARGS} ${MODEL_ARGS} ${DATA_ARGS} ${WANDB_ARGS}


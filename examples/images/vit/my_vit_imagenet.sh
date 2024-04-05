set -xe
# pip install -r requirements.txt

dataset="imagenet"
DATA_ARGS="
  --data_dir /media/dataset1/ImageNet2012 \
  --dataset ${dataset} \
  --num_class 1000 \
  --img_size 224 \
"

PLUGIN="hybrid_parallel"  # plugin(training strategy): "torch_ddp"/"torch_ddp_fp16"/"low_level_zero"/"gemini"/"hybrid_parallel"
tp_size=1
PARALLEL_ARGS="
  --plugin ${PLUGIN} \
  --tp_size ${tp_size} \
  --pp_size 1 \
"

lr=3e-3
lr_scheduler="linear"   # 'linear', 'cosine'
batch_size=128
TRAINING_ARGS="
  --num_epoch 200 \
  --batch_size ${batch_size} \
  --lr_scheduler ${lr_scheduler} \
  --learning_rate ${lr} \
  --weight_decay 0.1 \
  --label_smoothing \
  --seed 42 \
"

model_name="vit_base"   # 'vit_tiny' 'vit_small' 'vit_base' 'vit_large'   
patch_size=32
MODEL_ARGS="
  --model_name ${model_name} \
  --patch_size ${patch_size} \
  --output_path ./output_model \
  --dropout_ratio 0.1 \
  --warmup_ratio 0.01 \
"

is_wandb="online"   # ['disabled', 'online']
WANDB_ARGS="
--is_wandb ${is_wandb} \
--project_name lignex1_vit_cifar100 \
--exp_name gpu01_tp${tp_size}_${dataset}_bs${batch_size}_${model_name}/${patch_size}_${lr_scheduler}lr${lr} \
--wandb_save_dir /media/dataset1/lig_jinlovespho/log/colAI \
"

# number of gpus to use
GPUNUM=2

DISTRIBUTED_ARGS="
  --standalone \
  --nproc_per_node ${GPUNUM} \
"

CUDA_VISIBLE_DEVICES=0,1 torchrun ${DISTRIBUTED_ARGS} my_vit_train.py ${TRAINING_ARGS} ${PARALLEL_ARGS} ${MODEL_ARGS} ${DATA_ARGS} ${WANDB_ARGS}


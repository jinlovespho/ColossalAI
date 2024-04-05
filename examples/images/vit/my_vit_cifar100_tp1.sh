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
tp_size=1
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

model_name="vit_splithead_small"   # 'vit_tiny' 'vit_small' 'vit_base' 'vit_large' 'vit_splithead_tiny/small/base'
patch_size=4
splithead_method=4    # 0:linear, 1:featurewise, 2:featureconv, 3:shuffle 4:roll
MODEL_ARGS="
  --model_name ${model_name} \
  --patch_size ${patch_size} \
  --output_path ./output_model \
  --dropout_ratio 0.1 \
  --warmup_ratio 0.01 \
  --splithead_method ${splithead_method} \
"

CUDA=1
is_wandb="online"   # ['disabled', 'online']
WANDB_ARGS="
--is_wandb ${is_wandb} \
--project_name lignex1_vit_cifar100 \
--exp_name gpu${CUDA}_tp${tp_size}_${dataset}_${lr_scheduler}lr${lr}_${model_name}${patch_size}_splithead${splithead_method} \
--wandb_save_dir /media/dataset1/lig_jinlovespho/log/colAI \
"

# number of gpus to use
GPUNUM=1

DISTRIBUTED_ARGS="
  --standalone \
  --nproc_per_node ${GPUNUM} \
"

CUDA_VISIBLE_DEVICES=${CUDA} torchrun ${DISTRIBUTED_ARGS} my_vit_train.py ${TRAINING_ARGS} ${PARALLEL_ARGS} ${MODEL_ARGS} ${DATA_ARGS} ${WANDB_ARGS}


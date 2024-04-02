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
TRAINING_ARGS="
  --num_epoch 200 \
  --batch_size 128 \
  --learning_rate ${lr} \
  --weight_decay 0.0001 \
  --grad_checkpoint True \
  --seed 42 \
"

model_name="vit_small"   # 'vit_tiny' 'vit_small' 'vit_base' 'vit_large'   
patch_size=4
MODEL_ARGS="
  --model_name ${model_name} \
  --patch_size ${patch_size} \
  --hidden_dropout_prob 0.0 \
  --attention_probs_dropout_prob 0.1 \
  --output_path ./output_model \
  --dropout_ratio 0.1 \
  --warmup_ratio 0.01 \
  --model_name_or_path google/vit-base-patch16-224 \
"

is_wandb="disabled"   # ['disabled', 'online']
WANDB_ARGS="
--is_wandb ${is_wandb} \
--project_name lignex1_vit_cifar100 \
--exp_name server05_gpu12_${dataset}_${model_name}${patch_size}_tp${tp_size}_lr${lr}_noOptimization_fp32 \
--wandb_save_dir /media/dataset1/lig_jinlovespho/log/colAI \
"

# number of gpus to use
GPUNUM=2

DISTRIBUTED_ARGS="
  --standalone \
  --nproc_per_node ${GPUNUM} \
"

CUDA_VISIBLE_DEVICES=1,2 torchrun ${DISTRIBUTED_ARGS} vit_train_demo.py ${TRAINING_ARGS} ${PARALLEL_ARGS} ${MODEL_ARGS} ${DATA_ARGS} ${WANDB_ARGS}
# CUDA_VISIBLE_DEVICES=5,6 colossalai run ${DISTRIBUTED_ARGS} --master_port 29505 vit_train_demo.py ${TRAINING_ARGS} ${PARALLEL_ARGS} ${MODEL_ARGS} ${DATA_ARGS} ${WANDB_ARGS}


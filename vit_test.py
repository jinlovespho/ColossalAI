from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from torchvision import datasets 
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam

SEED = 42
MODEL_PATH = "google/vit-base-patch16-224"
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.0
NUM_EPOCH = 3
WARMUP_RATIO = 0.3
TP_SIZE = 2
PP_SIZE = 2

# Launch ColossalAI
colossalai.launch_from_torch(config={})
coordinator = DistCoordinator()
world_size = coordinator.world_size


image_processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
# train_dataset = BeansDataset(image_processor, TP_SIZE, split="train")
# eval_dataset = BeansDataset(image_processor, RP_SIZE, split="validation")
data_path = '/media/dataset1/jinlovespho_ds/cifar100'
train_dataset = datasets.CIFAR100(root=data_path, download=True, train=True )
val_dataset = datasets.CIFAR100(root=data_path, download=True, train=False )
num_labels = train_dataset.num_labels

train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)


config = ViTConfig.from_pretrained(MODEL_PATH)
config.num_labels = num_labels
config.id2label = {str(i): c for i, c in enumerate(train_dataset.label_names)}
config.label2id = {c: str(i) for i, c in enumerate(train_dataset.label_names)}
model = ViTForImageClassification.from_pretrained(
    MODEL_PATH, config=config, ignore_mismatched_sizes=True
)

optimizer = HybridAdam(model.parameters(), lr=(LEARNING_RATE * world_size), weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

total_steps = len(train_dataloader) * NUM_EPOCH
num_warmup_steps = int(WARMUP_RATIO * total_steps)
lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer, total_steps=(len(train_dataloader) * NUM_EPOCH), warmup_steps=num_warmup_steps
    )

def _criterion(outputs, inputs):
    return outputs.loss

plugin = HybridParallelPlugin(
            tp_size=TP_SIZE,
            pp_size=PP_SIZE,
            num_microbatches=None,
            microbatch_size=1,
            enable_all_optimization=True,
            precision="fp16",
            initial_scale=1,
        )
booster = Booster(plugin=plugin)

model, optimizer, _criterion, train_dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, criterion=criterion, dataloader=train_dataloader, lr_scheduler=lr_scheduler
    )


def run_forward_backward(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: Callable[[Any, Any], torch.Tensor],
    data_iter: Iterator,
    booster: Booster,
):
    if optimizer is not None:
        optimizer.zero_grad()
    if isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1:
        # run pipeline forward backward when enabling pp in hybrid parallel plugin
        output_dict = booster.execute_pipeline(
            data_iter, model, criterion, optimizer, return_loss=True, return_outputs=True
        )
        loss, outputs = output_dict["loss"], output_dict["outputs"]
    else:
        batch = next(data_iter)
        outputs = model(**batch)
        loss = criterion(outputs, None)
        if optimizer is not None:
            booster.backward(loss, optimizer)

def train_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    criterion: Callable[[Any, Any], torch.Tensor],
    lr_scheduler: LRScheduler,
    dataloader: DataLoader,
    booster: Booster,
    coordinator: DistCoordinator,
):
    torch.cuda.synchronize()

    num_steps = len(dataloader)
    data_iter = iter(dataloader)
    enable_pbar = coordinator.is_master()
    if isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1:
        # when using pp, only the last stage of master pipeline (dp_rank and tp_rank are both zero) shows pbar
        tp_rank = dist.get_rank(booster.plugin.tp_group)
        dp_rank = dist.get_rank(booster.plugin.dp_group)
        enable_pbar = tp_rank == 0 and dp_rank == 0 and booster.plugin.stage_manager.is_last_stage()
    model.train()

    with tqdm(range(num_steps), desc=f"Epoch [{epoch + 1}]", disable=not enable_pbar) as pbar:
        for _ in pbar:
            loss, _ = run_forward_backward(model, optimizer, criterion, data_iter, booster)
            optimizer.step()
            lr_scheduler.step()

            # Print batch loss
            if enable_pbar:
                pbar.set_postfix({"loss": loss.item()})
                

for epoch in range(NUM_EPOCH):
    train_epoch(epoch, model, optimizer, criterion, lr_scheduler, train_dataloader, booster, coordinator)
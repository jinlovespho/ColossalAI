from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from args import parse_demo_args

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR, LinearWarmupLR
from colossalai.nn.optimizer import HybridAdam


# JINLOVESPHO
import sys
import pdb
from util.utils import get_dataset, get_model, get_criterion
import wandb
import numpy as np
from thop import profile
from torchprofile import profile_macs 
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table
import os 
import time
from torch.cuda import nvtx

# ForkedPdb().set_trace()
class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def move_to_cuda(batch, device):
    # cifar100
    batch[0] = batch[0].to(device)
    batch[1] = batch[1].to(device)
    return batch


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
        # breakpoint()
        batch = next(data_iter)
        batch = move_to_cuda(batch, torch.cuda.current_device())
        outputs = model(batch[0])
        # ForkedPdb().set_trace()
        
        loss = criterion(outputs, batch[1])
        if optimizer is not None:
            booster.backward(loss, optimizer)

    return loss, outputs


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

    train_avg_loss_lst=[]
    with tqdm(range(num_steps), desc=f"Epoch [{epoch + 1}]", disable=not enable_pbar) as pbar:
        for _ in pbar:
            # breakpoint()
            torch.cuda.synchronize()
            start_time = time.time()
            
            loss, _ = run_forward_backward(model, optimizer, criterion, data_iter, booster)
            torch.cuda.synchronize()
            end_time = time.time()
            inf_time=end_time-start_time
            throughput = 1/inf_time 
            
            wandb.log( {'throughput':throughput} )
            wandb.log( {'inference_time':inf_time} )
            
            optimizer.step()
            lr_scheduler.step()
            
            train_avg_loss_lst.append(loss.item())
            wandb.log( {'train_step_loss':loss} )

            # Print batch loss
            if enable_pbar:
                pbar.set_postfix({"loss": loss.item()})
            
            lr = lr_scheduler.get_last_lr()
            wandb.log( {'lr':lr[0]} )

                
    # ForkedPdb().set_trace()     
    train_avg_loss = np.mean(train_avg_loss_lst)
    wandb.log( {'train_avg_loss':train_avg_loss } )
    

@torch.no_grad()
def evaluate_model(
    epoch: int,
    model: nn.Module,
    criterion: Callable[[Any, Any], torch.Tensor],
    eval_dataloader: DataLoader,
    booster: Booster,
    coordinator: DistCoordinator,
):
    torch.cuda.synchronize()
    model.eval()
    accum_loss = torch.zeros(1, device=torch.cuda.current_device())
    total_num = torch.zeros(1, device=torch.cuda.current_device())
    accum_correct = torch.zeros(1, device=torch.cuda.current_device())

    # batch[0]: data
    # batch[1]: label
    for batch in eval_dataloader:
        # breakpoint()
        batch = move_to_cuda(batch, torch.cuda.current_device())
        loss, outputs = run_forward_backward(model, None, criterion, iter([batch]), booster)

        to_accum = True
        if isinstance(booster.plugin, HybridParallelPlugin):
            # when using hybrid parallel, loss is only collected from last stage of pipeline with tp_rank == 0
            to_accum = to_accum and (dist.get_rank(booster.plugin.tp_group) == 0)
            if booster.plugin.pp_size > 1:
                to_accum = to_accum and booster.plugin.stage_manager.is_last_stage()

        if to_accum:
            accum_loss += loss / len(eval_dataloader)
            logits = outputs
            preds = torch.argmax(logits, dim=1)

            labels = batch[1]
            total_num += batch[1].shape[0]
            accum_correct += torch.sum(preds == labels)
                
    # ForkedPdb().set_trace()
    dist.all_reduce(accum_loss)
    dist.all_reduce(total_num)
    dist.all_reduce(accum_correct)
    avg_loss = "{:.4f}".format(accum_loss.item())
    accuracy = "{:.4f}".format(accum_correct.item() / total_num.item())
    
    # breakpoint()
    wandb.log({ 'val_loss':float(avg_loss),
                'val_acc':float(accuracy),
                'epoch':epoch+1 })

    if coordinator.is_master():
        print(
            f"Evaluation result for epoch {epoch + 1}: \
                average_loss={avg_loss}, \
                accuracy={accuracy}."
        )

def main():
    # breakpoint()
    args = parse_demo_args()

    # Launch ColossalAI
    colossalai.launch_from_torch(config={}, seed=args.seed)
    coordinator = DistCoordinator()
    world_size = coordinator.world_size
    print('world size: ', world_size)
    print('tp_size: ', args.tp_size)
    print('pp_size: ', args.pp_size)
    
    # ForkedPdb().set_trace()
    
    if coordinator.is_master():
        print('master')
        print(coordinator.local_rank)
    else:
        print('not master')
        print(coordinator.local_rank)
    
    # Manage loggers
    disable_existing_loggers()
    logger = get_dist_logger()
    if coordinator.is_master():
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # Reset tp_size and pp_size to 1 if not using hybrid parallel.
    if args.plugin != "hybrid_parallel":
        args.tp_size = 1
        args.pp_size = 1

    # Prepare Dataset
    train_ds, val_ds = get_dataset(args)
    
    # breakpoint()
    
    model = get_model(args)
    logger.info(f"Finish loading model from {args.model_name_or_path}", ranks=[0])


    # Set plugin
    booster_kwargs = {}
    if args.plugin == "torch_ddp_fp16":
        booster_kwargs["mixed_precision"] = "fp16"
    if args.plugin.startswith("torch_ddp"):
        plugin = TorchDDPPlugin()
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(offload_optim_frac=1.0, pin_memory=True, initial_scale=2**5)
    elif args.plugin == "low_level_zero":
        plugin = LowLevelZeroPlugin(initial_scale=2**5)
    elif args.plugin == "hybrid_parallel":
        plugin = HybridParallelPlugin(
            tp_size=args.tp_size,
            pp_size=args.pp_size,
            zero_stage=0,   # normal torch ddp 
            num_microbatches=None,
            microbatch_size=1,
            enable_all_optimization=False,
            precision="fp32",
            initial_scale=1,
        )
    else:
        raise ValueError(f"Plugin with name {args.plugin} is not supported!")
    logger.info(f"Set plugin as {args.plugin}", ranks=[0])

    # Prepare dataloader
    train_dataloader = plugin.prepare_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    eval_dataloader = plugin.prepare_dataloader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    # Set optimizer
    optimizer = HybridAdam(model.parameters(), lr=(args.learning_rate * world_size), weight_decay=args.weight_decay)
    criterion = get_criterion(args)

    # Set lr scheduler
    total_steps = len(train_dataloader) * args.num_epoch
    num_warmup_steps = int(args.warmup_ratio * total_steps)
    if args.lr_scheduler == 'linear':
        lr_scheduler = LinearWarmupLR(optimizer=optimizer, total_steps=(len(train_dataloader) * args.num_epoch), warmup_steps=num_warmup_steps)
    elif args.lr_scheduler =='cosine':
        lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer, total_steps=(len(train_dataloader) * args.num_epoch), warmup_steps=num_warmup_steps)

    # breakpoint()
    # Set booster
    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, criterion, train_dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, criterion=criterion, dataloader=train_dataloader, lr_scheduler=lr_scheduler
    )
    
    # total number of parameters
    args.module_params = sum(i.numel() for i in model.module.parameters())
    args.model_params = sum(i.numel() for i in model.parameters())
    
    # macs, flops 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # test_input = torch.rand(1,3,args.img_size,args.img_size).to(device)
    # args.macs_torchprofile = profile_macs(model.module, test_input )
    # args.macs_thop, args.thop_params = profile(model.module, inputs=(test_input, ))
    # flops_fvcore = FlopCountAnalysis(model.module, test_input)
    
    # flop_count_table(args.flops_fvcore)
    # parameter_count_table(model.module)
    
    print('=='*70)
    print(args.module_params)
    print(args.model_params)
    print(f'DATASET: {args.dataset}, LR_SCHEDULER: {args.lr_scheduler}, VIT_MODEL: {args.model_name},  splithead_method: {args.splithead_method}')
    print(f'DATASET: {args.dataset}, LR_SCHEDULER: {args.lr_scheduler}, VIT_MODEL: {args.model_name},  splithead_method: {args.splithead_method}')
    print('=='*70)
    
    # ForkedPdb().set_trace()
    
    # setup wandb logger
    wandb.init( 
            project=args.project_name,
            name=args.exp_name,
            config=args,
            dir=args.wandb_save_dir,
            mode = args.is_wandb
            )
    
    # Finetuning
    logger.info(f"Start Training", ranks=[0])
    for epoch in range(args.num_epoch):
        # breakpoint()
        nvtx.range_push('model forward_split')
        train_epoch(epoch, model, optimizer, criterion, lr_scheduler, train_dataloader, booster, coordinator)
        nvtx.range_pop()
        # ForkedPdb().set_trace()    
        evaluate_model(epoch, model, criterion, eval_dataloader, booster, coordinator)
    logger.info(f"Finish finetuning", ranks=[0])

    # save model & optimizer
    save_path = f'{args.output_path}/{args.exp_name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
          
    save_dict = {
        'model_state': model.module.state_dict(),
        'opt_state':optimizer.state_dict(),
        'num_epoch':args.num_epoch,
    }
    
    torch.save(save_dict, f'{save_path}.pth')
    
    # Save the finetuned model
    booster.save_model(model, args.output_path, shard=True)
    logger.info(f"Saving model checkpoint to {args.output_path}", ranks=[0])

if __name__ == "__main__":
    main()
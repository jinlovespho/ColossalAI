import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from util.autoaugment import CIFAR10Policy, SVHNPolicy
from util.criterions import LabelSmoothingCrossEntropyLoss
from util.da import RandomCropPaste

def get_criterion(args):
    if args.criterion=="ce":
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(args.num_classes, smoothing=args.smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"{args.criterion}?")

    return criterion

def get_model(args):
    if args.model_name == 'vit':
        from networks.vit import ViT
        breakpoint()
        net = ViT(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token
            )
        
    # JINLOVESPHO
    elif args.model_name == 'vit_tiny_8':
        from networks.vit import ViT
        breakpoint()
        net = ViT(
            in_c = 3,
            num_classes = args.num_class,
            img_size = args.img_size ,
            patch = args.patch_size,
            dropout = args.dropout_ratio,
            num_layers = 12,
            hidden = 192,
            mlp_hidden=192*4,
            head=3,
            is_cls_token=True
        )
    
    # JINLOVESPHO
    elif args.model_name == 'vit_parallel':
        from networks.vit_parallel import ViT_Parallel
        breakpoint()
        net = ViT_Parallel(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token
            )  
        
    # JINLOVESPHO
    elif args.model_name == 'vit_gyu':
        from networks.vit_gyu import ViT_Gyu
        breakpoint()
        net = ViT_Gyu(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token
            )
    
    elif args.model_name == 'vit-Tiny_crossVit':
        from networks.vit_tiny_crossvit import ViT_Tiny_CrossVit
        breakpoint()
        net = ViT_Tiny_CrossVit(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token
            )
    
    elif args.model_name == 'vit-Small_crossVit':
        from networks.vit_small_crossvit import ViT_Small_CrossVit
        breakpoint()
        net = ViT_Small_CrossVit(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token
            )        
    
    else:
        raise NotImplementedError(f"{args.model_name} is not implemented yet...")

    return net

def get_transform(**kwargs):
    train_transform = []
    test_transform = []
    train_transform += [
        transforms.RandomCrop(size=kwargs['size'], padding=kwargs['padding'])
    ]
     
    train_transform.append(CIFAR10Policy())
    
    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=kwargs['mean'], std=kwargs['std'])
    ]
    
    test_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=kwargs['mean'], std=kwargs['std'])
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform
    

def get_dataset(data_dir):
    
    kwargs={'in_c':3, 'num_classes':100, 'size':32, 'padding':4, 'mean':[0.5071, 0.4867, 0.4408], 'std':[0.2675, 0.2565, 0.2761]}
    train_transform, test_transform = get_transform(**kwargs)
    train_ds = torchvision.datasets.CIFAR100(data_dir, train=True, transform=train_transform, download=True)
    val_ds = torchvision.datasets.CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    
    return train_ds, val_ds

def get_experiment_name(args):
    experiment_name = f"{args.experiment_memo}"
    if args.autoaugment:
        experiment_name+="_aa"
    if args.label_smoothing:
        experiment_name+="_ls"
    if args.rcpaste:
        experiment_name+="_rc"
    if args.cutmix:
        experiment_name+="_cm"
    if args.mixup:
        experiment_name+="_mu"
    if args.off_cls_token:
        experiment_name+="_gap"
    print(f"Experiment:{experiment_name}")
    return experiment_name

import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets

from util.autoaugment import CIFAR10Policy, SVHNPolicy, ImageNetPolicy
from util.criterions import LabelSmoothingCrossEntropyLoss
from util.da import RandomCropPaste


def get_criterion(args):
    if args.criterion=="ce":
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(args.num_class, smoothing=0.1)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"{args.criterion}?")

    return criterion


def get_model(args):
    # breakpoint()
    if args.model_name == 'vit_tiny':
        from networks.vit import ViT
        # breakpoint()
        print('vit_tiny')
        args.num_layers=12
        args.hidden=192
        args.mlp_hidden=768
        args.head=3
        net = ViT(
            in_c=3,
            num_classes=args.num_class, 
            img_size=args.img_size,
            patch_size=args.patch_size, 
            dropout=args.dropout_ratio, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=True
            )
        
    # JINLOVESPHO
    elif args.model_name == 'vit_small':
        from networks.vit import ViT
        # breakpoint()
        print('vit_small')
        args.num_layers=12
        args.hidden=384
        args.mlp_hidden=1536
        args.head=6
        net = ViT(
            in_c=3,
            num_classes=args.num_class, 
            img_size=args.img_size,
            patch_size=args.patch_size, 
            dropout=args.dropout_ratio, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=True
            )
    
    # JINLOVESPHO
    elif args.model_name == 'vit_base':
        from networks.vit import ViT
        # breakpoint()
        print('vit_base')
        args.num_layers=12
        args.hidden=768
        args.mlp_hidden=3072
        args.head=12
        net = ViT(
            in_c=3,
            num_classes=args.num_class, 
            img_size=args.img_size,
            patch_size=args.patch_size, 
            dropout=args.dropout_ratio, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=True
            )
        
    # JINLOVESPHO
    elif args.model_name == 'vit_large':
        from networks.vit import ViT
        # breakpoint()
        print('vit_large')
        args.num_layers=24
        args.hidden=1024
        args.mlp_hidden=4096
        args.head=16
        net = ViT(
            in_c=3,
            num_classes=args.num_class, 
            img_size=args.img_size,
            patch_size=args.patch_size, 
            dropout=args.dropout_ratio, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=True
            )
    
    # JINLOVESPHO
    elif args.model_name == 'vit_huge':
        from networks.vit import ViT
        # breakpoint()
        print('vit_huge')
        args.num_layers=32
        args.hidden=1280
        args.mlp_hidden=5120
        args.head=16
        net = ViT(
            in_c=3,
            num_classes=args.num_class, 
            img_size=args.img_size,
            patch_size=args.patch_size, 
            dropout=args.dropout_ratio, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=True
            )    
    
    # JINLOVESPHO
    elif args.model_name == 'vit_splithead_tiny':
        from networks.vit_splithead import ViT_SplitHead
        # breakpoint()
        print('vit_splithead_tiny')
        args.num_layers=12
        args.hidden=192
        args.mlp_hidden=768
        args.head=3
        net = ViT_SplitHead(
            in_c=3,
            num_classes=args.num_class, 
            img_size=args.img_size,
            patch_size=args.patch_size, 
            dropout=args.dropout_ratio, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=True,
            splithead_method=args.splithead_method
            )
        
    # JINLOVESPHO
    elif args.model_name == 'vit_splithead_small':
        from networks.vit_splithead import ViT_SplitHead
        # breakpoint()
        print('vit_splithead_small')
        args.num_layers=12
        args.hidden=384
        args.mlp_hidden=1536
        args.head=6
        net = ViT_SplitHead(
            in_c=3,
            num_classes=args.num_class, 
            img_size=args.img_size,
            patch_size=args.patch_size, 
            dropout=args.dropout_ratio, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=True,
            splithead_method=args.splithead_method
            )
    
    # JINLOVESPHO
    elif args.model_name == 'vit_splithead_base':
        from networks.vit_splithead import ViT_SplitHead
        # breakpoint()
        print('vit_splithead_base')
        args.num_layers=12
        args.hidden=768
        args.mlp_hidden=3072
        args.head=12
        net = ViT_SplitHead(
            in_c=3,
            num_classes=args.num_class, 
            img_size=args.img_size,
            patch_size=args.patch_size, 
            dropout=args.dropout_ratio, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=True,
            splithead_method=args.splithead_method
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

def get_transform(args):
    train_transform = []
    test_transform = []
    
    if args.dataset == 'cifar100':
        train_transform += [transforms.RandomCrop(size=args.size, padding=args.padding)] 
        train_transform += [transforms.RandomHorizontalFlip()]
        train_transform.append(CIFAR10Policy())
        train_transform += [transforms.ToTensor(), transforms.Normalize(mean=args.mean, std=args.std) ]
        
        test_transform += [transforms.ToTensor(), transforms.Normalize(mean=args.mean, std=args.std)]

    elif args.dataset == 'imagenet':
        mean, std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform += [transforms.Resize((args.img_size, args.img_size))]
        train_transform += [transforms.RandomCrop(size=args.crop_size, padding=4, pad_if_needed=False)]
        train_transform += [transforms.RandomHorizontalFlip()]
        train_transform.append(ImageNetPolicy())
        
        train_transform += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]   
        test_transform += [transforms.Resize((args.img_size, args.img_size))]
        test_transform += [transforms.RandomCrop(size=args.crop_size, padding=4, pad_if_needed=True)]          
        test_transform += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform

def get_dataset(args):
    if args.dataset == 'cifar100':
        # breakpoint()
        print('DATASET: cifar100')
        args.in_c = 3
        args.num_classes=100
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        train_ds = datasets.CIFAR100(args.data_dir, train=True, transform=train_transform, download=True)
        val_ds = datasets.CIFAR100(args.data_dir, train=False, transform=test_transform, download=True)

    elif args.dataset == 'imagenet':
        # breakpoint()
        print('DATASET: imagenet')
        args.num_classes = 1000
        args.crop_size = 224
        train_transform, val_transform = get_transform(args)
        train_ds = datasets.ImageNet(root=args.data_dir, split='train', transform=train_transform )
        val_ds = datasets.ImageNet(root=args.data_dir, split='val', transform=val_transform )
            
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

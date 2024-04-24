import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import numpy as np
import argparse
from torchvision import models
from torchvision import ops
from torchvision.models.detection import rpn
from torchvision.models.detection import FasterRCNN

from coco_dataset import COCODataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim

from tqdm import tqdm
from utils import *
from make_args import Args
from train_func import train

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default='./config/apple2orange_single_gpu.json', help="config path")
opt = parser.parse_args()

# load config.json
args = Args(opt.config_path)

# dataloader
def collator(batch):
    return tuple(zip(*batch))

transform = transforms.Compose(
    [
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(dtype=torch.float)
    ]
)

train_dataset = COCODataset(args.data_path, train=True, transform=transform)
val_dataset = COCODataset(args.data_path, train=False, transform=transform)

train_dataloader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collator, num_workers=args.num_workers
)
val_dataloader = DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collator, num_workers=args.num_workers
)

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.backbone == 'vgg16':
    backbone = models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1").features
    backbone.out_channels = 512

    anchor_generator = rpn.AnchorGenerator(sizes=(args.anchor_sizes,), aspect_ratios=(args.anchor_ratio,))
    roi_pooler = ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=args.pooler_output_size, sampling_ratio=args.pooler_sampling_ratio)

    model = FasterRCNN(backbone=backbone, 
                       num_classes=len(train_dataset._get_categories()), 
                       rpn_anchor_generator=anchor_generator, 
                       box_roi_pool=roi_pooler ).to(device)
    
elif args.backbone == 'resnet50fpn':
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    model = FasterRCNN(backbone, num_classes=len(train_dataset._get_categories())).to('cuda')

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# train
train(args, model, train_dataloader, val_dataloader, optimizer)
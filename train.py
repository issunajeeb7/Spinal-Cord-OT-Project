import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from importlib import import_module
from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry
from trainer import trainer_spinalcord
from icecream import ic
import nni

# --- MODIFIED: Import class configuration ---
from config import NUM_CLASSES, CLASSES

hyp_params = {
    "batch": 16,
    "lr": 0.005,
    "epochs": 200
}

optimized_params = nni.get_next_parameter()
hyp_params.update(optimized_params)

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default=r'C:\Users\issu\Documents\OT\segmentation_dataset\segmentation dataset', help='root dir for data')
parser.add_argument('--output', type=str, default='./output/samed_spinalcord')
parser.add_argument('--dataset', type=str,
                    default='SpinalCord', help='experiment_name')
# --- REMOVED: num_classes is now loaded from config.py ---
parser.add_argument('--max_epochs', type=int,
                    default=hyp_params['epochs'], help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int,
                    default=160, help='early stopping epoch')
parser.add_argument('--batch_size', type=int,
                    default=hyp_params['batch'], help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=hyp_params['lr'],
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_h', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='checkpoints\sam_vit_h_4b8939.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument('--lr_exp', type=float, default=0.9, help='The learning rate decay expotential')
parser.add_argument('--tf32', action='store_true', default=True, help='If activated, use tf32 to accelerate the training process')
parser.add_argument('--compile', action='store_true', default=False, help='If activated, compile the training model for acceleration')
parser.add_argument('--use_amp', action='store_true', default=False, help='If activated, adopt mixed precision for acceleration')

args = parser.parse_args()

if __name__ == "__main__":
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if not args.deterministic:
        cudnn.benchmark = True
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    
    dataset_config = {
        'SpinalCord': {
            'root_path': args.root_path,
            # --- MODIFIED: Use NUM_CLASSES from config ---
            'num_classes': NUM_CLASSES,
        }
    }
    args.is_pretrain = True
    args.exp = dataset_name + '_' + str(args.img_size)
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_s' + str(args.seed)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # register model
    # --- MODIFIED: Use NUM_CLASSES from config ---
    sam = sam_model_registry[args.vit_name](checkpoint=args.ckpt)

    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank, args.img_size, NUM_CLASSES).cuda()
    if args.compile:
        net = torch.compile(net)

    if args.lora_ckpt is not None:
        net.load_lora_parameters(args.lora_ckpt)

    multimask_output = NUM_CLASSES > 1
    embedding_grid = net.sam.prompt_encoder.image_embedding_size[0]
    low_res = embedding_grid * 4

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')
    # Also save the class configuration to the output folder
    config_items.append(f'CLASSES: {",".join(CLASSES)}\n')


    with open(config_file, 'w') as f:
        f.writelines(config_items)

    trainer = {'SpinalCord': trainer_spinalcord}
    trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res, NUM_CLASSES)
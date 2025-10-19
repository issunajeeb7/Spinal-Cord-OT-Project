import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry
from dataset_spinalcord import SpinalCordDataset

# --- MODIFIED: Import class configuration ---
from config import CLASSES, NUM_CLASSES

def inference(args, multimask_output, db_config, model, test_save_path=None):
    db_test = db_config['Dataset'](base_dir=args.root_path, split='test')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label = sampled_batch['image'], sampled_batch['label']
        case_name = f"test_case_{i_batch}"
        
        metric_i = test_single_volume(image, label, model, classes=NUM_CLASSES, multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=db_config['z_spacing'])
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)[0]))
            
    metric_list = metric_list / len(db_test)
    # --- MODIFIED: Dynamically print class names from config ---
    for i in range(1, NUM_CLASSES):  # Start from 1 to skip background
        try:
            class_name = CLASSES[i]
            logging.info('Mean class %d name %s mean_dice %f' % (i, class_name, metric_list[i-1][0]))
        except IndexError:
            logging.info('Mean class %d mean_dice %f' % (i, metric_list[i-1][0]))
            
    performance = np.mean(metric_list, axis=0)[0]
    logging.info('Testing performance: mean_dice : %f' % (performance))
    logging.info("Testing Finished!")
    return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default=r'C:\Users\issu\Documents\OT\segmentation_dataset\segmentation dataset', help='root dir for data')
    parser.add_argument('--dataset', type=str, default='SpinalCord', help='Experiment name')
    parser.add_argument('--output_dir', type=str, default='./output/samed_spinalcord')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='../checkpoints/sam_vit_h_4b8939.pth',
                        help='Pretrained SAM checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='./output/samed_spinalcord/SpinalCord_224_epo200_bs16_lr0.005_s1234/best_model.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_h', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

    args = parser.parse_args()

    if not args.deterministic:
        cudnn.benchmark = True
    else:
        cudnn.benchmark = False
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    dataset_name = args.dataset
    dataset_config = {
        'SpinalCord': {
            'Dataset': SpinalCordDataset,
            'root_path': args.root_path,
            # --- MODIFIED: Use NUM_CLASSES from config ---
            'num_classes': NUM_CLASSES,
            'z_spacing': 1
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # --- MODIFIED: Use NUM_CLASSES from config ---
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=NUM_CLASSES,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None, "Please provide the path to your trained LoRA checkpoint."
    net.load_lora_parameters(args.lora_ckpt)

    multimask_output = NUM_CLASSES > 1

    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
        
    inference(args, multimask_output, dataset_config[dataset_name], net, test_save_path)
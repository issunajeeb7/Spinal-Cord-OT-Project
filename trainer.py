import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from icecream import ic
import nni

# --- MODIFIED: Import our new dataset loader ---
from dataset_spinalcord import SpinalCordDataset, RandomGenerator


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    if low_res_label_batch.shape[-2:] != low_res_logits.shape[-2:]:
        low_res_label_batch = F.interpolate(
            low_res_label_batch.unsqueeze(1).float(),
            size=low_res_logits.shape[-2:],
            mode='nearest'
        ).squeeze(1)

    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


# --- MODIFIED: Renamed function for clarity ---
def trainer_spinalcord(args, model, snapshot_path, multimask_output, low_res, num_classes):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    batch_size = args.batch_size * args.n_gpu

    # --- MODIFIED: Create separate datasets for training and validation ---
    db_train = SpinalCordDataset(base_dir=args.root_path, split="train",
                               transform=RandomGenerator(output_size=[args.img_size, args.img_size]),
                               low_res_size=(low_res, low_res))

    db_val = SpinalCordDataset(base_dir=args.root_path, split="val",
                             transform=RandomGenerator(output_size=[args.img_size, args.img_size]),
                             low_res_size=(low_res, low_res))

    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val set is: {}".format(len(db_val)))


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                           worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    # --- IMPORTANT: Ensure DiceLoss gets the correct number of classes ---
    dice_loss = DiceLoss(num_classes)

    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            image_batch = image_batch.float() / 255.0
            low_res_label_batch = low_res_label_batch.cuda()

            if iter_num % 10 == 0:
                logging.info(
                    "[Debug] epoch=%d iter=%d batch=%d image_device=%s image_shape=%s low_res_shape=%s",
                    epoch_num,
                    iter_num,
                    i_batch,
                    image_batch.device,
                    tuple(image_batch.shape),
                    tuple(low_res_label_batch.shape),
                )

            if args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    outputs = model(image_batch, multimask_output, args.img_size)
                    loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                outputs = model(image_batch, multimask_output, args.img_size)
                loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
            else:
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** args.lr_exp

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

        # --- Validation Loop ---
        model.eval()
        val_dice_score = 0
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                low_res_label_batch = sampled_batch['low_res_label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                low_res_label_batch = low_res_label_batch.cuda()

                outputs = model(image_batch, multimask_output, args.img_size)
                # Calculate validation loss for monitoring
                loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
                
                # You might want a more direct validation metric like IoU or Dice on full-res masks
                # This part can be expanded with metrics from utils.py

        model.train()
        
        # This is a placeholder for a real validation metric.
        # For NNI, you should calculate a metric like mIoU or average Dice score here.
        current_performance = 1.0 - loss_dice.item() # Using 1-DiceLoss as a simple metric
        nni.report_intermediate_result(float(current_performance))

        if current_performance > best_performance:
            best_performance = current_performance
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info(f"Saved best model to {save_mode_path}")

        if (epoch_num + 1) % 20 == 0:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info(f"Saved model to {save_mode_path}")

        if epoch_num >= stop_epoch -1:
            logging.info("Reached stop epoch. Ending training.")
            iterator.close()
            break
            
    nni.report_final_result(float(best_performance))
    writer.close()
    return "Training Finished!"
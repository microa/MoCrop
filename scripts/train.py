


import os
import random
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision
from torch.utils.data import DataLoader
import cv2
import argparse

# ============ Key Dependency: Assumes dataset_all.py exists ============
from src.mocrop_dataset import MoCropDataset

# --- 1. Data Transformation Classes (Unchanged) ---
class GroupMultiScaleCrop(object):
    def __init__(self, input_size, scales=None, max_distort=1):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.input_size = [input_size, input_size] if not isinstance(input_size, list) else input_size
    def __call__(self, img_group):
        im_size = img_group[0].shape
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img[offset_h:offset_h + crop_h, offset_w:offset_w + crop_w] for img in crop_img_group]
        ret_img_group = [cv2.resize(img, (self.input_size[0], self.input_size[1]), cv2.INTER_LINEAR) for img in crop_img_group]
        return ret_img_group
    def _sample_crop_size(self, im_size):
        image_h, image_w, _ = im_size
        base_size = min(image_h, image_w)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]
        pairs = [(w, h) for h in crop_h for w in crop_w if abs(w / (h + 1e-6) - 1) <= self.max_distort]
        crop_pair = random.choice(pairs)
        w_offset = random.randint(0, image_w - crop_pair[0])
        h_offset = random.randint(0, image_h - crop_pair[1])
        return crop_pair[0], crop_pair[1], w_offset, h_offset

class GroupRandomHorizontalFlip(object):
    def __call__(self, img_group):
        if random.random() < 0.5:
            return [img[:, ::-1, :] for img in img_group]
        else:
            return img_group

class GroupScale(object):
    def __init__(self, size):
        self._size = (size, size)
    def __call__(self, img_group):
        return [cv2.resize(img, self._size, cv2.INTER_LINEAR) for img in img_group]

# --- 2. Utility Functions (Unchanged) ---
def color_aug(img, random_h=36, random_l=50, random_s=50):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(float)
    h = (random.random() * 2 - 1.0) * random_h
    l = (random.random() * 2 - 1.0) * random_l
    s = (random.random() * 2 - 1.0) * random_s
    img[..., 0] = np.minimum(img[..., 0] + h, 180)
    img[..., 1] = np.minimum(img[..., 1] + l, 255)
    img[..., 2] = np.minimum(img[..., 2] + s, 255)
    img = np.maximum(img, 0)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HLS2BGR)
    return img

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay):
    decay = 0
    for i in range(len(lr_steps)):
        if epoch >= lr_steps[i]:
            decay += 1
    lr = args.lr * (lr_decay ** decay)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# --- 3. Model Definition (Updated to support all Table 1 models) ---
class VideoBackboneModel(nn.Module):
    def __init__(self, num_class, num_segments, base_model='resnet50'):
        super(VideoBackboneModel, self).__init__()
        self.num_segments = num_segments
        print(f"Initializing model with base: {base_model}, num_class: {num_class}, num_segments: {num_segments}.")
        model_map = {
            'resnet50': (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.DEFAULT),
            'mobilenet_v3_large': (torchvision.models.mobilenet_v3_large, torchvision.models.MobileNet_V3_Large_Weights.DEFAULT),
            'efficientnet_b1': (torchvision.models.efficientnet_b1, torchvision.models.EfficientNet_B1_Weights.DEFAULT),
            'swin_t': (torchvision.models.swin_t, torchvision.models.Swin_T_Weights.DEFAULT),
            'swin_b': (torchvision.models.swin_b, torchvision.models.Swin_B_Weights.DEFAULT),
        }
        if base_model not in model_map:
            raise ValueError(f"Unsupported base model: '{base_model}'.")
        model_constructor, model_weights = model_map[base_model]
        self.base_model = model_constructor(weights=model_weights)
        if 'resnet' in base_model:
            feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(feature_dim, num_class)
        elif 'mobilenet_v3' in base_model or 'efficientnet' in base_model:
            feature_dim = self.base_model.classifier[-1].in_features
            self.base_model.classifier[-1] = nn.Linear(feature_dim, num_class)
        elif 'swin' in base_model:
            feature_dim = self.base_model.head.in_features
            self.base_model.head = nn.Linear(feature_dim, num_class)
        else:
            raise NotImplementedError(f"Classifier modification for {base_model} is not implemented.")
    def forward(self, x):
        x = x.view((-1,) + x.size()[-3:])
        base_out = self.base_model(x)
        return base_out

# --- 4. Training and Validation Functions (Unchanged) ---
def train(train_loader, model, criterion, optimizer, epoch, cur_lr, num_segments, print_freq=100, alpha=0.2):
    model.train()
    batch_time, data_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    end = time.time()
    for i, (input_, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_ = input_.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        input_mixed, target_a, target_b, lam = mixup_data(input_, target, alpha=alpha)
        output = model(input_mixed)
        output = torch.mean(output.view(target.size(0), num_segments, -1), dim=1)
        loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), target.size(0))
        top1.update(prec1, target.size(0))
        top5.update(prec5, target.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]\\t"
                  f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\\t"
                  f"Data {data_time.val:.3f} ({data_time.avg:.3f})\\t"
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})\\t"
                  f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\\t"
                  f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})")
    print(f"[Training Results]: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {losses.avg:.5f}")

def validate(val_loader, model, criterion, num_segments, print_freq=100):
    model.eval()
    batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        for i, (input_, target) in enumerate(val_loader):
            input_ = input_.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(input_)
            output = torch.mean(output.view(target.size(0), num_segments, -1), dim=1)
            loss = criterion(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), target.size(0))
            top1.update(prec1, target.size(0))
            top5.update(prec5, target.size(0))
            if i % print_freq == 0:
                print(f"Test: [{i}/{len(val_loader)}]\\t"
                      f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\\t"
                      f"Loss {losses.val:.4f} ({losses.avg:.4f})\\t"
                      f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\\t"
                      f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})")
    print(f"[Validation Results]: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {losses.avg:.5f}")
    return top1.avg

# --- 5. Main Function ---
def main():
    parser = argparse.ArgumentParser(description="PyTorch Video Action Recognition Training")
    # --- Core Parameters ---
    parser.add_argument('--arch', default='efficientnet_b1',
                        choices=['resnet50', 'mobilenet_v3_large', 'efficientnet_b1', 'swin_t', 'swin_b'],
                        help='model architecture')
    parser.add_argument('--input-size', default=224, type=int, help='size of input images')
    parser.add_argument('--num-segments', default=3, type=int, help='number of segments for training')

    # --- Training Strategy Parameters ---
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--optimizer', default='adamw', type=str.lower, choices=['sgd', 'adam', 'adamw'],
                        help='optimizer type')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--lr-steps', default=[40, 80], type=int, nargs='+', help='epochs to decay learning rate')
    parser.add_argument('--lr-decay', default=0.1, type=float, help='learning rate decay factor')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (L2 penalty)')

    # --- Validation and MoCrop Parameters ---
    parser.add_argument('--val-segments', default=25, type=int, help='number of segments for validation')
    parser.add_argument('--crop-ratio', default=0.0, type=float, help='crop ratio for MoCrop. Set to 0.0 to disable.')
    parser.add_argument('--mv-h', default=6, type=int, help='motion vector grid height for MoCrop')
    parser.add_argument('--mv-w', default=8, type=int, help='motion vector grid width for MoCrop')
    
    #  New Parameter to control training mode 
    parser.add_argument('--train-mode', default='normal', choices=['normal', 'mocrop'],
                        help='Use normal or mocrop data for training.')

    # --- Miscellaneous ---
    parser.add_argument('--eval-freq', default=1, type=int, help='evaluation frequency (in epochs)')
    parser.add_argument('-j', '--workers', default=8, type=int, help='data loading workers')
    parser.add_argument('--print-freq', default=150, type=int, help='print frequency')

    args = parser.parse_args()

    # --- Data Loading ---
    if args.train_mode == 'mocrop':
        print("Using MoCrop training mode with crop_ratio={}, mv_h={}, mv_w={}".format(args.crop_ratio, args.mv_h, args.mv_w))
        train_dataset = MoCropDataset(
            data_root='/home/mbin/data/ucf101/mpeg4_videos',
            video_list='/home/mbin/data/datalists/ucf101_split1_train.txt',
            num_segments=args.num_segments,
            is_train=True,
            transform=GroupMultiScaleCrop(args.input_size) + GroupRandomHorizontalFlip() + GroupScale(args.input_size),
            representation='iframe',
            accumulate=False,
            mv_h=args.mv_h,
            mv_w=args.mv_w,
            crop_ratio=args.crop_ratio
        )
    else:
        print("Using normal training mode")
        train_dataset = MoCropDataset(
            data_root='/home/mbin/data/ucf101/mpeg4_videos',
            video_list='/home/mbin/data/datalists/ucf101_split1_train.txt',
            num_segments=args.num_segments,
            is_train=True,
            transform=GroupMultiScaleCrop(args.input_size) + GroupRandomHorizontalFlip() + GroupScale(args.input_size),
            representation='iframe',
            accumulate=False
        )

    val_dataset = MoCropDataset(
        data_root='/home/mbin/data/ucf101/mpeg4_videos',
        video_list='/home/mbin/data/datalists/ucf101_split1_test.txt',
        num_segments=args.val_segments,
        is_train=False,
        transform=GroupScale(args.input_size),
        representation='iframe',
        accumulate=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # --- Model Setup ---
    model = VideoBackboneModel(num_class=101, num_segments=args.num_segments, base_model=args.arch)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Training Loop ---
    best_prec1 = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps, args.lr_decay)
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}, lr={cur_lr}")
        
        train(train_loader, model, criterion, optimizer, epoch, cur_lr, args.num_segments)
        
        if epoch % args.eval_freq == 0:
            prec1 = validate(val_loader, model, criterion, args.val_segments)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            
            if is_best:
                arch_short_map = {'resnet50': 'Res50', 'mobilenet_v3_large': 'MNv3L', 'efficientnet_b1': 'EB1', 'swin_t': 'SwinT', 'swin_b': 'SwinB'}
                arch_short = arch_short_map[args.arch]
                mode_suffix = 'mocrop' if args.train_mode == 'mocrop' else 'normal'
                save_path = f"{arch_short}_ucf101_i{args.input_size}_s{args.num_segments}_val-{mode_suffix}_opt-{args.optimizer}_best.pth.tar"
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()

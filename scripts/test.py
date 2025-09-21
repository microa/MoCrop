import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import cv2
import argparse

from src.mocrop_dataset import MoCropDataset

class VideoBackboneModel(nn.Module):
    def __init__(self, num_class, num_segments, base_model='resnet50'):
        super(VideoBackboneModel, self).__init__()
        self.num_segments = num_segments
        print(f"Initializing model with base: {base_model}, num_class: {num_class}, num_segments: {num_segments}.")
        
        model_map = {
            'resnet50': (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.DEFAULT),
            'resnet152': (torchvision.models.resnet152, torchvision.models.ResNet152_Weights.DEFAULT), 
            'mobilenet_v3_large': (torchvision.models.mobilenet_v3_large, torchvision.models.MobileNet_V3_Large_Weights.DEFAULT),
            'efficientnet_b1': (torchvision.models.efficientnet_b1, torchvision.models.EfficientNet_B1_Weights.DEFAULT),
            'swin_t': (torchvision.models.swin_t, torchvision.models.Swin_T_Weights.DEFAULT),
            'swin_b': (torchvision.models.swin_b, torchvision.models.Swin_B_Weights.DEFAULT), 
        }

        if base_model not in model_map:
            raise ValueError(f"Unsupported base model: '{base_model}'. Supported models are: {list(model_map.keys())}")
            
        model_constructor, model_weights = model_map[base_model]
        self.backbone = model_constructor(weights=model_weights)

        if 'resnet' in base_model:
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(feature_dim, num_class)
        elif 'mobilenet_v3' in base_model or 'efficientnet' in base_model:
            feature_dim = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(feature_dim, num_class)
        elif 'swin' in base_model:
            feature_dim = self.backbone.head.in_features
            self.backbone.head = nn.Linear(feature_dim, num_class)
        else:
            raise NotImplementedError(f"Classifier modification for {base_model} is not implemented.")

    def forward(self, x):
        x = x.view((-1,) + x.size()[-3:])
        base_out = self.backbone(x)
        return base_out

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

def main():
    parser = argparse.ArgumentParser(description='Video Action Recognition Testing')
    parser.add_argument('--weights', required=True, help='path to model weights')
    parser.add_argument('--arch', default='mobilenet_v3_large',
                        choices=['resnet50', 'resnet152', 'mobilenet_v3_large', 'efficientnet_b1', 'swin_t', 'swin_b'],
                        help='model architecture')
    parser.add_argument('--input-size', default=224, type=int, help='size of input images')
    parser.add_argument('--test-segments', default=8, type=int, help='number of segments for testing')
    parser.add_argument('--test-mode', default='normal', choices=['normal', 'mocrop'],
                        help='testing mode: normal or mocrop')
    parser.add_argument('--crop-ratio', default=0.75, type=float, help='crop ratio for MoCrop')
    parser.add_argument('--mv-h', default=6, type=int, help='motion vector grid height')
    parser.add_argument('--mv-w', default=8, type=int, help='motion vector grid width')
    parser.add_argument('--batch-size', default=16, type=int, help='batch size for testing')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    
    args = parser.parse_args()

    config = {
        'data_name': 'ucf101',
        'data_root': '/home/mbin/data/ucf101/mpeg4_videos',
        'test_list': '/home/mbin/data/datalists/ucf101_split1_test.txt',
        'weights': args.weights,
        'arch': args.arch,
        'input_size': args.input_size,
        'test_segments': args.test_segments,
        'test_mode': args.test_mode,
        'crop_ratio': args.crop_ratio if args.test_mode == 'mocrop' else None,
        'mv_h': args.mv_h if args.test_mode == 'mocrop' else None,
        'mv_w': args.mv_w if args.test_mode == 'mocrop' else None,
        'workers': args.workers,
        'batch_size': args.batch_size
    }

    print("--- Testing with the following configuration ---")
    for key, value in config.items():
        display_value = value if value is not None else "N/A (Not Applicable in normal mode)"
        print(f"{key:<15}: {display_value}")
    print("-------------------------------------------------")

    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Loading checkpoint from: {args.weights}")
    
    try:
        checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    except:
        checkpoint = torch.load(args.weights, map_location='cpu')

    model = VideoBackboneModel(num_class=101, num_segments=args.test_segments, base_model=args.arch)
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
        elif k.startswith('backbone.'):
            name = k[9:]
        elif k.startswith('base_model.'):
            name = k[11:]
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.cuda()
    model.eval()
    print("Model loaded successfully.\n")

    if args.test_mode == 'mocrop':
        print(f"INFO: Using MoCrop testing with crop_ratio={args.crop_ratio}, mv=({args.mv_h}, {args.mv_w})")
        test_dataset = MoCropDataset(
            data_root=config['data_root'],
            video_list=config['test_list'],
            num_segments=args.test_segments,
            is_train=False,
            transform=None,
            representation='iframe',
            accumulate=False,
            mv_h=args.mv_h,
            mv_w=args.mv_w,
            crop_ratio=args.crop_ratio
        )
    else:
        print("INFO: Using normal testing.")
        test_dataset = MoCropDataset(
            data_root=config['data_root'],
            video_list=config['test_list'],
            num_segments=args.test_segments,
            is_train=False,
            transform=None,
            representation='iframe',
            accumulate=False
        )

    from torch.utils.data import DataLoader
    data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    total_num = len(test_dataset)
    print(f"Starting testing on {total_num} videos...")
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (input_tensor, target) in enumerate(data_loader):
            input_tensor = input_tensor.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            output = model(input_tensor)
            output = torch.mean(output.view(target.size(0), args.test_segments, -1), dim=1)
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if i % 50 == 0:
                print(f'  ...Processed batch {i + 1}/{len(data_loader)}')
    
    accuracy = 100. * correct / total
    print(f'\n Testing completed!\nAccuracy: {accuracy:.3f}% ({correct}/{total})')
    
    if isinstance(checkpoint, dict) and 'best_prec1' in checkpoint:
        print(f"Best accuracy recorded during training: {checkpoint['best_prec1']:.3f}%")

if __name__ == '__main__':
    main()

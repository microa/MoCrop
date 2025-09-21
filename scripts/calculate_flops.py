import torch
import torch.nn as nn
import torchvision
from fvcore.nn import FlopCountAnalysis
import logging

class VideoBackboneModel(nn.Module):
    def __init__(self, num_class, num_segments, base_model='resnet50'):
        super(VideoBackboneModel, self).__init__()
        self.num_segments = num_segments
        
        model_map = {
            'resnet50': (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.DEFAULT),
            'mobilenet_v3_large': (torchvision.models.mobilenet_v3_large, torchvision.models.MobileNet_V3_Large_Weights.DEFAULT),
            'efficientnet_b1': (torchvision.models.efficientnet_b1, torchvision.models.EfficientNet_B1_Weights.DEFAULT),
            'swin_b': (torchvision.models.swin_b, torchvision.models.Swin_B_Weights.DEFAULT), 
            'resnet152': (torchvision.models.resnet152, torchvision.models.ResNet152_Weights.DEFAULT), 
        }

        if base_model not in model_map:
            raise ValueError(f"Unsupported base model: '{base_model}'. Supported models are: {list(model_map.keys())}")
            
        model_constructor, model_weights = model_map[base_model]
        self.base_model = model_constructor(weights=None) 

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

def main():
    archs_to_analyze = ['resnet50', 'resnet152', 'mobilenet_v3_large', 'efficientnet_b1', "swin_b"]
    normal_input_size = 224
    mocrop_input_sizes = [192, 224]
    num_class = 101 
    num_segments = 8

    fvcore_logger = logging.getLogger('fvcore')
    fvcore_logger.setLevel(logging.ERROR)

    print("-" * 80)
    print(f"{'Model Architecture':<25} | {'Mode':<10} | {'Input Size':<12} | {'FLOPs (GFLOPs)':<15}")
    print("-" * 80)

    for arch in archs_to_analyze:
        model_wrapper = VideoBackboneModel(num_class, num_segments, base_model=arch)
        model_to_analyze = model_wrapper.base_model
        model_to_analyze.eval()

        inputs_normal = torch.randn(1, 3, normal_input_size, normal_input_size)
        flop_analyzer_normal = FlopCountAnalysis(model_to_analyze, inputs_normal)
        flops_normal = flop_analyzer_normal.total() / 1e9

        print(f"{arch:<25} | {'Normal':<10} | {f'{normal_input_size}x{normal_input_size}':<12} | {flops_normal:<15.2f}")

        for size in mocrop_input_sizes:
            inputs_mocrop = torch.randn(1, 3, size, size)
            flop_analyzer_mocrop = FlopCountAnalysis(model_to_analyze, inputs_mocrop)
            flops_mocrop = flop_analyzer_mocrop.total() / 1e9

            print(f"{arch:<25} | {'MoCrop':<10} | {f'{size}x{size}':<12} | {flops_mocrop:<15.2f}")
        
        print("-" * 80)

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import cv2

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

def create_model(num_class, num_segments, base_model='resnet50'):
    return VideoBackboneModel(num_class, num_segments, base_model)

class GroupMultiScaleCrop(object):
    def __init__(self, input_size, scales=None, max_distort=1):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.input_size = [input_size, input_size] if not isinstance(input_size, list) else input_size
    def __call__(self, img_group):
        im_size = img_group[0].shape
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img[offset_h:offset_h + crop_h, offset_w:offset_w + crop_w] for img in img_group]
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

def color_augmentation(img, random_h=36, random_l=50, random_s=50):
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

def apply_multiple_augmentations(image, augmentation_list):
    augmented_images = {}
    for name, transform in augmentation_list:
        augmented_image = transform(image)
        augmented_images[name] = augmented_image
    
    return augmented_images

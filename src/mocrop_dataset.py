import os
import random
import numpy as np
import torch
import torch.utils.data as data
import cv2


def convert_to_yolo_format(top_left_x, top_left_y, bottom_right_x, bottom_right_y, img_width, img_height):
    center_x = (top_left_x + bottom_right_x) / 2.0
    center_y = (top_left_y + bottom_right_y) / 2.0

    width = bottom_right_x - top_left_x
    height = bottom_right_y - top_left_y

    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height

    return center_x, center_y, width, height


def mv2yolo(n, area_ratio=0.8, img_width=320, img_height=240):
    rows, cols = n.shape
    cell_height = img_height // rows
    cell_width = img_width // cols

    max_sum = -np.inf
    best_top_left = (0, 0)
    best_bottom_right = (0, 0)

    target_area = area_ratio * rows * cols
    for h in range(1, rows + 1):
        for w in range(1, cols + 1):
            if h * w > target_area * 1.1 or h * w < target_area * 0.9:
                continue
            for i in range(rows - h + 1):
                for j in range(cols - w + 1):
                    current_sum = np.sum(n[i:i+h, j:j+w])
                    if current_sum > max_sum:
                        max_sum = current_sum
                        best_top_left = (i, j)
                        best_bottom_right = (i + h - 1, j + w - 1)

    top_left_x = best_top_left[1] * cell_width
    top_left_y = best_top_left[0] * cell_height
    bottom_right_x = (best_bottom_right[1] + 1) * cell_width
    bottom_right_y = (best_bottom_right[0] + 1) * cell_height

    center_x, center_y, width, height = convert_to_yolo_format(
        top_left_x, top_left_y, bottom_right_x, bottom_right_y, img_width, img_height
    )
    return center_x, center_y, width, height


def crop_image_with_motion_vectors(img, npy, area_ratio):
    bbox = mv2yolo(npy, area_ratio)

    if img is None:
        raise FileNotFoundError(f"Image not found or unable to read: {img}")

    if len(bbox) != 4:
        raise ValueError(f"bbox should contain exactly 4 elements: [x_center, y_center, width, height]. Received: {bbox}")

    img_height, img_width = img.shape[:2]

    x_center_norm, y_center_norm, width_norm, height_norm = bbox

    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height
    width = width_norm * img_width
    height = height_norm * img_height

    x1 = int(round(x_center - (width / 2)))
    y1 = int(round(y_center - (height / 2)))
    x2 = int(round(x_center + (width / 2)))
    y2 = int(round(y_center + (height / 2)))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)

    if x1 >= x2 or y1 >= y2:
        print(f"Warning: Invalid crop coordinates calculated: ({x1}, {y1}), ({x2}, {y2}). Returning original image.")
        return img

    cropped_img = img[y1:y2, x1:x2]
    return cropped_img


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


def get_num_frames(video_path):
    """Get number of frames in video using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def load_video_frame(video_path, frame_index):
    """Load a specific frame from video using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    return frame


class MoCropDataset(data.Dataset):
    def __init__(self, data_root, video_list, num_segments, is_train, transform,
                 representation='iframe', accumulate=False,
                 mv_h=None, mv_w=None, crop_ratio=None):

        self.data_root = data_root
        self.video_list_file = video_list
        self.num_segments = num_segments
        self.is_train = is_train
        self.transform = transform
        
        self.representation = representation
        self.accumulate = accumulate
        self.mv_h = mv_h
        self.mv_w = mv_w
        self.crop_ratio = crop_ratio

        self._input_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).float()
        self._input_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).float()

        self._video_records = []
        self._load_list()

    def _load_list(self):
        self._video_records = []
        with open(self.video_list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                video, label = parts[0], parts[-1]
                video_path = os.path.join(self.data_root, video[:-4] + '.mp4')
                try:
                    numf = get_num_frames(video_path)
                    if numf > self.num_segments:
                         self._video_records.append((video_path, int(label), numf))
                except Exception as e:
                    print(f"Warning: Could not get frame count for {video_path}, skipping. Error: {e}")
        print(f"Loaded {len(self._video_records)} videos from {self.video_list_file}.")

    def _get_frame_index(self, num_frames, seg):
        seg_size = float(num_frames - 1) / self.num_segments
        if self.is_train:
            seg_begin = int(np.round(seg_size * seg))
            seg_end = int(np.round(seg_size * (seg + 1)))
            if seg_end == seg_begin:
                seg_end = seg_begin + 1
            v_frame_idx = random.randint(seg_begin, seg_end - 1)
        else:
            v_frame_idx = int(np.round(seg_size * (seg + 0.5)))
        
        v_frame_idx = max(0, min(v_frame_idx, num_frames - 1))
        return v_frame_idx
    
    def get_npy_path(self, video_path):
      video_filename = os.path.basename(video_path).split('.')[0]
      category = os.path.basename(os.path.dirname(video_path))
      npy_dir = os.path.join('/home/mbin/data/ucf101/extract_mvs', category, video_filename)
      npy_filename = f'motion_vectors_denoised_all_mc_{self.mv_h}_{self.mv_w}.npy'
      return os.path.join(npy_dir, npy_filename)

    def __getitem__(self, index):
        if self.is_train and index < len(self._video_records):
            video_path, label, num_frames = random.choice(self._video_records)
        elif index < len(self._video_records):
            video_path, label, num_frames = self._video_records[index]
        else:
            return self.__getitem__(index % len(self._video_records))

        frames = []
        for seg in range(self.num_segments):
            frame_index = self._get_frame_index(num_frames, seg)
            
            img = load_video_frame(video_path, frame_index)

            if img is None:
                print(f"Warning: loading video {video_path} frame failed. Using blank image.")
                img = np.zeros((256, 256, 3), dtype=np.uint8)
                frames.append(img)
                continue

            if self.crop_ratio is not None and self.crop_ratio > 0:
                npy_path = self.get_npy_path(video_path)
                if os.path.exists(npy_path):
                    npy = np.load(npy_path)
                    img = crop_image_with_motion_vectors(img, npy, self.crop_ratio)
                else:
                    print(f'Warning: npy file not found at {npy_path}. Using original img.')
            
            if self.is_train:
                img = color_augmentation(img)
            
            img = img[..., ::-1]
            frames.append(img)

        if self.transform is not None:
            frames = self.transform(frames)
            
        frames_np = np.array(frames)
        tensor = torch.from_numpy(np.transpose(frames_np, (0, 3, 1, 2))).float() / 255.0
        tensor = (tensor - self._input_mean) / self._input_std
        
        return tensor, label

    def __len__(self):
        return len(self._video_records)

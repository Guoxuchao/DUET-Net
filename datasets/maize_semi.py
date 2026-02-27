from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import scipy.io as scio
import scipy.spatial

def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area


class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train', info=None):

        self.root_path = root_path
        # Support both jpg and png formats
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')) + glob(os.path.join(self.root_path, '*.png')))
        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")
        self.label_list = []
        if method in ['train']:
            try:
                label_list_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'label_list')
                with open(os.path.join(label_list_dir, info+'.txt')) as f:
                    for i in f:
                        self.label_list.append(i.strip())
            except:
                raise Exception(f"please give right info, failed to load {os.path.join(label_list_dir, info+'.txt')}")

        self.method = method
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
        
        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        # Strong Augmentation for Unlabeled Data
        # Note: Geometric transforms (Flip/Crop) are shared and applied before this in __getitem__
        # Here we add Random Color Jitter with 80% probability
        self.strong_aug = transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ], p=0.8)

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = os.path.splitext(img_path)[0] + '.npy'
        
        if not os.path.exists(gd_path):
            mat_path = os.path.splitext(img_path)[0] + '.mat'
            if os.path.exists(mat_path):
                mat_data = scio.loadmat(mat_path)
                
                if 'annPoints' in mat_data:
                    keypoints = mat_data['annPoints']
                elif 'image_info' in mat_data:
                     keypoints = mat_data['image_info'][0,0][0,0][0]
                else:
                    found = False
                    for key in mat_data:
                        if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                            keypoints = mat_data[key]
                            found = True
                            break
                    if not found:
                         raise Exception(f"Cannot find keypoints in {mat_path}")
                
                np.save(gd_path, keypoints)
            else:
                 raise FileNotFoundError(f"Ground truth file not found for {img_path}. Expected .npy or .mat")

        img = Image.open(img_path).convert('RGB')
        
        if self.method == 'train':
            keypoints = np.load(gd_path)
            label = (os.path.basename(img_path) in self.label_list)
            return self.train_transform(img, keypoints, label)
            
        elif self.method == 'val' or self.method == 'test':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, keypoints, name

    def train_transform(self, img, keypoints, label):
        """Random crop, geometric transforms, and augmentation."""
        wd, ht = img.size
        
        # Random resize
        re_size = random.random() * 0.5 + 0.75
        wdd = (int)(wd * re_size)
        htt = (int)(ht * re_size)
        if min(wdd, htt) >= self.c_size:
            wd = wdd
            ht = htt
            img = img.resize((wd, ht))
            keypoints = keypoints * re_size
            
        st_size = min(wd, ht)
        assert st_size >= self.c_size
        
        # Random crop
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        
        if len(keypoints) > 0:
            # If keypoints lack depth/sigma info, estimate it using KNN
            if keypoints.shape[1] < 3:
                tree = scipy.spatial.KDTree(keypoints.copy(), leafsize=1024)
                distances, _ = tree.query(keypoints, k=4)
                avg_dist = np.mean(distances[:, 1:4], axis=1) 
                avg_dist = np.clip(avg_dist, 4.0, 40.0)
                keypoints = np.concatenate([keypoints, avg_dist[:, None]], axis=1)

            nearest_dis = np.clip(keypoints[:, 2], 4.0, 40.0)

            points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
            points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
            bbox = np.concatenate((points_left_up, points_right_down), axis=1)
            inner_area = cal_innner_area(j, i, j + w, i + h, bbox)
            origin_area = nearest_dis * nearest_dis
            ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
            mask = (ratio >= 0.5)

            target = ratio[mask]
            keypoints = keypoints[mask]
            keypoints = keypoints[:, :2] - [j, i]  

        # Random horizontal flip
        if random.random() > 0.5:
            img = F.hflip(img)
            if len(keypoints) > 0:
                keypoints[:, 0] = w - keypoints[:, 0]
        
        if len(keypoints) == 0:
            target = np.array([])
            
        # Weak Augmentation (Standard)
        img_weak = self.trans(img)
        
        # For Unlabeled data: Return both Weak and Strong views
        if not label:
            # Strong Augmentation (Color Jitter on top of geometric transforms)
            img_strong_pil = self.strong_aug(img)
            img_strong = self.trans(img_strong_pil)
            
            return (img_weak, img_strong), torch.from_numpy(keypoints.copy()).float(), \
                   torch.from_numpy(target.copy()).float(), st_size, label
        else:
            # For Labeled data: Return only Weak view
            return img_weak, torch.from_numpy(keypoints.copy()).float(), \
                   torch.from_numpy(target.copy()).float(), st_size, label

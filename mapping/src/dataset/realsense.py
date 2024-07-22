import os.path as osp

import cv2
import numpy as np
import torch
import csv
from glob import glob
from torch.utils.data import Dataset


class DataLoader(Dataset):
    def __init__(self, data_path, use_gt=False, max_depth=-1) -> None:
        self.data_path = data_path
        self.num_imgs = len(glob(osp.join(self.data_path, "results/*.png")))
        self.max_depth = max_depth
        self.use_gt = use_gt
        self.K = self.load_intrinsic()
        self.gt_pose = self.load_gt_pose()
        self.init = False

    def load_intrinsic(self):
        K = np.eye(3)
        K[0, 0] = 601.347290039062
        K[1, 1] = 601.343017578125
        K[0, 2] = 329.519226074219
        K[1, 2] = 238.586654663086

        return K

    def get_init_pose(self, init_frame=None):
        if self.gt_pose is not None and init_frame is not None:
            return self.gt_pose[init_frame].reshape(4, 4)
        elif self.gt_pose is not None:
            return self.gt_pose[0].reshape(4, 4)
        else:
            return np.eye(4)

    def load_gt_pose(self):
        gt_file = osp.join(self.data_path, 'traj.txt')
        gt_pose = np.loadtxt(gt_file) # (n_imgs,16)
        return gt_pose

    def load_depth(self, index):
        depth = cv2.imread(
            osp.join(self.data_path, 'results/depth{:06d}.png'.format(index)), cv2.IMREAD_UNCHANGED)
        depth = depth * 0.001
        if self.max_depth > 0:
            depth[(depth > self.max_depth)] = 0
        return depth

    def load_image(self, index):
        rgb = cv2.imread(
            osp.join(self.data_path, 'results/frame{:06d}.jpg'.format(index)), -1)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return rgb

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        img_raw = self.load_image(index)
        img = torch.from_numpy(img_raw / 255).float()
        depth = self.load_depth(index)
        depth = None if depth is None else torch.from_numpy(depth).float()  # (H,W)
        pose = self.gt_pose[index] if (self.use_gt or self.init == False) else None
        if self.init == False:
            self.init = True
        return index, img, depth, self.K, pose, img_raw


if __name__ == '__main__':
    import sys
    loader = DataLoader(sys.argv[1])
    for data in loader:
        index, img, depth, K, _ = data
        print(K)
        print(index, img.shape)
        cv2.imshow('img', img.numpy())
        cv2.imshow('depth', depth.numpy())
        cv2.waitKey(1)
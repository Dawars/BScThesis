import os
import sys
import pickle

import matplotlib.pyplot as plt
from skimage import io
from torch.utils.data import Dataset
import torch
from utils.image_utils import *
from config import args

sys.path.append('./src')


class HandJointDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, pkl_file, root_dir, size=224, pix_format='NHWC'):
        """
        Args:
            pkl_file (string): Path to the pkl file with annotations.
            root_dir (string): Directory with all the images.
        """
        with open(pkl_file, 'rb') as f:
            self.joints = pickle.load(f)

        self.root_dir = root_dir
        self.crop_size = size

    def __len__(self):
        return len(self.joints)

    def __getitem__(self, idx):
        data = self.joints[idx]
        filename_ = data['filename']
        img_name = os.path.join(self.root_dir, filename_)

        img = io.imread(img_name)
        keypoints = np.array(data['joints'])  # joint_id: x, y, certainty

        # crop
        cropped, keypoints = crop_img(img, keypoints)

        # resize
        resized, keypoints = resize_image(cropped, keypoints, size=(self.crop_size, self.crop_size))

        # flip horizontally
        resized = rotate_image_180(resized)
        keypoints[:, :2] = args.crop_size - 1 - keypoints[:, :2]

        if data['side'] == 'left':  # we only want right hand samples
            resized = flip_image_vertical(resized)
            keypoints[:, 0] = args.crop_size - 1 - keypoints[:, 0]

        # normalize kp to [-1, 1]
        ratio = 1.0 / args.crop_size
        keypoints[:, :2] = 2.0 * keypoints[:, :2] * ratio - 1.0
        dst_image = normalize_image(resized)

        # reorder joints and remove finger ends
        order = [0,  # wrist
                 5, 6, 7,  # index
                 9, 10, 11,  # middle
                 17, 18, 19,  # pinky
                 13, 14, 15,  # ring
                 1, 2, 3]  # thumb

        # set metacarpal certainty low - it is a 'virtual' joint
        keypoints[1, 2] = 0.2

        sample = {'image': torch.tensor(hwc2chw(dst_image), dtype=torch.float),
                  'joints': torch.tensor(keypoints[order, :]).float()}
        return sample


if __name__ == '__main__':
    dataset = HandJointDataset('/home/dawars/datasets/11k/11k_joints.pkl', '/home/dawars/datasets/11k/Raw', size=299)

    sample = dataset[0]
    img = sample['image'].numpy()
    joints = sample['joints'].numpy()

    img = chw2hwc(img)

    img = denormalize_image(img)

    # workaround layout error with cv circle flip image vertically
    img = cv2.flip(cv2.flip(img, 0), 0)

    joints[:, :2] = args.crop_size * (joints[:, :2] + 1.) / 2.

    img_joints = draw_joints(img, joints)

    plt.imshow(img_joints)
    plt.savefig('fig.png')

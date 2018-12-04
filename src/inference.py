"""Run the trained model in inference mode"""

import glob
import os
import pickle

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

from dataloader.hand_joint_dataloader import HandJointDataset
from utils.image_utils import *

from config import crop_size
from model import HMRNetBase
from util import draw_lsp_14kp__bone, copy_state_dict
from skimage import io


def inference(model, img_name, save_folder, crop_size=224):
    os.makedirs(save_folder, exist_ok=True)

    img = io.imread(img_name)[:, :, :3]
    dst_image = torch.tensor(hwc2chw(normalize_image(img)), dtype=torch.float)

    mano = model.mano

    src_image = dst_image.view(1, 3, crop_size, crop_size).cuda()

    generator_outputs = model(src_image)
    joint_preds = []
    for i, output in enumerate(generator_outputs):
        (theta, verts, j2d, j3d, Rs) = output
        joint_preds.append(j2d[0])

        vert = verts.detach().cpu().numpy()
        save_obj_name = os.path.basename(img_name).split('.')[0] + f"_{i}.obj"
        mano.save_obj(vert[0], os.path.join(save_folder, save_obj_name))

    strip = plot_joints(src_image[0], joint_preds)

    save_image_name = os.path.basename(img_name).split('.')[0] + ".png"

    io.imsave(os.path.join(save_folder, save_image_name), strip)


def plot_joints(image, joint_pred, act_joint=None):
    """
    Plot phases of iterative estimation
    :param image: path
    :param joint_pred:
    :param act_joint:
    :return:
    """
    images = []

    img = denormalize_image(chw2hwc(image.clone().cpu().detach().numpy())).astype(np.uint8)

    for pred in joint_pred:
        pred = image.shape[1] * (pred[:, :2] + 1) / 2
        img_pred = draw_joints(img, pred, color=(255, 0, 0))

        if act_joint is not None:
            actual_joints = image.shape[1] * (act_joint[:, :2] + 1) / 2
            img_pred = draw_joints(img_pred, actual_joints, color=(0, 0, 255))
        images.append(img_pred)

    return concat_images(images)


if __name__ == '__main__':
    model = HMRNetBase().cuda()
    model.train(False)
    w_visible = np.ones([16, 1], dtype=np.float)
    copy_state_dict(
        model.state_dict(),
        torch.load("/mnt/dawars/hdd1/model_saves/hand_10/10000_441.65061712265015_generator.pkl", map_location='cpu'),
        prefix='module.'
    )

    files = [
        'tests/hand1_dorsal.png',
        'tests/hand1_palm.png',
        'tests/hand2_dorsal.png',
        'tests/hand2_palm.png',
    ]
    for file in files:
        inference(model, file, '/mnt/dawars/hdd1/model_saves/hand_10/david_test10')

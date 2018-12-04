"""
    file:   trainer.py

    date:   2018_05_07
    author: zhangxiong(1025679612@qq.com)
"""

import sys

from dataloader.hand_joint_dataloader import HandJointDataset
from model import HMRNetBase
from config import args
import config
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tensorboardX import SummaryWriter

from util import align_by_pelvis, batch_rodrigues, copy_state_dict
import datetime
from collections import OrderedDict
import os

from utils.image_utils import *


class HMRTrainer(object):
    """
    HMR trainer.
    From an image input, trained a model that outputs 61D latent vector
    consisting of [cam (3 - [scale, tx, ty]), pose (48), shape (10)]
    """

    def __init__(self):
        self.pix_format = 'NCHW'
        self.normalize = True
        self.w_smpl = torch.ones(config.args.eval_batch_size).float().cuda()

        self.writer = SummaryWriter("/mnt/dawars/hdd1/model_saves/hand_10", "resnet_adv")

        self._build_model()
        self._create_data_loader()

    def _create_data_loader(self):
        self.loader_2d = self._create_2d_data_loader(config.train_2d_set)
        self.loader_eval = self._create_eval_data_loader(config.eval_set)

    def _build_model(self):
        print('start building model.')

        '''
            load pre-trained model
        '''
        generator = HMRNetBase()
        model_path = config.pre_trained_model['generator']
        if os.path.exists(model_path):
            copy_state_dict(
                generator.state_dict(),
                torch.load(model_path),
                prefix='module.'
            )
        else:
            print('model {} does not exist!'.format(model_path))

        self.generator = torch.nn.DataParallel(generator).cuda()

        # encoder optimizer (e_)
        self.e_opt = torch.optim.Adam(
            self.generator.parameters(),
            lr=args.e_lr,
            weight_decay=args.e_wd
        )

        self.e_sche = torch.optim.lr_scheduler.StepLR(
            self.e_opt,
            step_size=500,
            gamma=0.9
        )

        print('finished build model.')

    def _create_2d_data_loader(self, data_2d_set):
        data_set = []
        for data_set_name in data_2d_set:
            data_set_path = config.data_set_path[data_set_name]
            if data_set_name == '11k_train':
                hand_train = HandJointDataset(
                    pkl_file=data_set_path,
                    root_dir='/home/dawars/datasets/11k/Raw',
                    pix_format=self.pix_format,
                    size=args.crop_size
                )
                data_set.append(hand_train)
            else:
                msg = 'invalid 2d dataset'
                sys.exit(msg)

        con_2d_dataset = ConcatDataset(data_set)
        print(f"Batch size: {config.args.batch_size}")
        return DataLoader(
            dataset=con_2d_dataset,
            batch_size=config.args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=config.args.num_worker
        )

    def _create_eval_data_loader(self, data_eval_set):
        data_set = []
        for data_set_name in data_eval_set:
            data_set_path = config.data_set_path[data_set_name]
            if data_set_name == '11k_val':
                hand_val = HandJointDataset(
                    pkl_file=data_set_path,
                    root_dir='/home/dawars/datasets/11k/Raw',
                    pix_format=self.pix_format,
                    size=args.crop_size
                )
                data_set.append(hand_val)
            else:
                msg = 'invalid eval dataset'
                sys.exit(msg)
        con_eval_dataset = ConcatDataset(data_set)
        return DataLoader(
            dataset=con_eval_dataset,
            batch_size=config.args.eval_batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=config.args.num_worker
        )

    def train(self):
        def save_model(result):
            exclude_key = 'module.mano'

            def exclude_mano(model_dict):
                result = OrderedDict()
                for (k, v) in model_dict.items():
                    if exclude_key in k or 'module.mano' in k:
                        continue
                    result[k] = v
                return result

            parent_folder = args.save_folder
            os.makedirs(parent_folder, exist_ok=True)

            title = result['title']
            generator_save_path = os.path.join(parent_folder, title + 'generator.pkl')
            torch.save(exclude_mano(self.generator.state_dict()), generator_save_path)
            with open(os.path.join(parent_folder, title + '.txt'), 'w') as fp:
                fp.write(str(result))

        torch.backends.cudnn.benchmark = True
        loader_2d = iter(self.loader_2d)
        loader_eval = iter(self.loader_eval)
        e_opt = self.e_opt

        self.generator.train()

        for iter_index in range(config.args.iter_count):
            try:
                data_2d = next(loader_2d)
            except StopIteration:  # restart dataset
                loader_2d = iter(self.loader_2d)
                data_2d = next(loader_2d)

            image_from_2d = data_2d['image']
            images = image_from_2d.cuda()

            generator_outputs = self.generator(images)

            loss_kp_2d, loss_shape = self._calc_loss(generator_outputs, data_2d)

            # 100 * kp_loss_2d + loss_shape + e_disc_loss
            e_loss = args.e_loss_weight * loss_kp_2d + loss_shape

            e_opt.zero_grad()
            e_loss.backward()
            e_opt.step()

            loss_kp_2d = float(loss_kp_2d)
            loss_shape = float(loss_shape / args.e_shape_ratio)

            e_loss = args.e_loss_weight * loss_kp_2d + loss_shape

            iter_msg = OrderedDict(
                [
                    ('time', datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')),
                    ('iter', iter_index),
                    ('e_loss', e_loss),
                    ('2d_loss', loss_kp_2d),
                    ('shape_loss', loss_shape),
                ]
            )
            print(iter_msg)

            if iter_index % 500 == 0:

                # tensorboardX log
                # evaluation

                try:
                    eval_batch = next(loader_eval)
                except StopIteration:  # restart dataset
                    loader_eval = iter(self.loader_eval)
                    eval_batch = next(loader_eval)

                image_from_2d = eval_batch['image']
                images = image_from_2d.cuda()

                generator_outputs = self.generator(images)

                eval_loss_kp_2d, eval_loss_shape = \
                    self._calc_loss(generator_outputs, eval_batch)

                eval_total_loss = args.e_loss_weight * eval_loss_kp_2d + eval_loss_shape

                self.writer.add_scalars('loss/2d_joint', {
                    'train': float(loss_kp_2d),
                    'eval': float(eval_loss_kp_2d)}, iter_index)
                self.writer.add_scalars('loss/shape_loss', {
                    'train': loss_shape,
                    'eval': float(eval_loss_shape)}, iter_index)
                self.writer.add_scalars('loss/total_loss', {
                    'train': e_loss,
                    'eval': float(eval_total_loss)}, iter_index)

                for i in range(6):
                    # images
                    joint_preds = []
                    for (theta, verts, j2d, j3d, Rs) in generator_outputs:
                        joint_preds.append(j2d[i])
                    strip = self.plot_joints(eval_batch['image'][i], eval_batch['joints'][i], joint_preds)
                    self.writer.add_image(f'img/joint_fitting_{i}', torch.tensor(hwc2chw(strip / 255)), iter_index)

                    # shape histogram
                    last_beta = generator_outputs[-1][0][i, 51:]
                    self.writer.add_histogram(f"params/shape_{i}", np.abs(last_beta.clone().cpu().data.numpy()),
                                              iter_index)

                iter_msg['title'] = '{}_{}_'.format(iter_msg['iter'], iter_msg['e_loss'])
                save_model(iter_msg)

    def plot_joints(self, image, act_joint, joint_pred):
        """
        Plot joint predictions on image
        :param image: sample image
        :param act_joint: joint annotations
        :param joint_pred: joint predictions
        :return: image strip for every iteration
        """
        images = []
        actual_joints = image.shape[1] * (act_joint[:, :2] + 1) / 2

        img = denormalize_image(chw2hwc(image.clone().cpu().detach().numpy())).astype(np.uint8)

        for pred in joint_pred:
            pred = image.shape[1] * (pred[:, :2] + 1) / 2
            img_pred = draw_joints(img, pred, color=(255, 0, 0))
            img_both = draw_joints(img_pred, actual_joints, color=(0, 0, 255))
            images.append(img_both)
        return concat_images(images)

    def _calc_loss(self, generator_outputs, data_2d):
        sample_2d_count = data_2d['joints'].shape[0]

        (predict_theta, predict_verts, predict_j2d, predict_j3d, predict_Rs) = generator_outputs[-1]

        real_2d = data_2d['joints'].cuda()
        predict_j2d, predict_j3d, predict_theta = predict_j2d, predict_j3d[sample_2d_count:, :], \
                                                  predict_theta[sample_2d_count:, :]

        loss_kp_2d = self.batch_kp_2d_l1_loss(real_2d, predict_j2d) * args.e_loss_weight

        # regularize shape params
        betas = generator_outputs[-1][0][:, 51:]

        shape_dif = betas ** 2
        loss_shape = shape_dif.sum(1).sum(0) / betas.shape[0]

        return loss_kp_2d, loss_shape

    def batch_kp_2d_l1_loss(self, real_2d_kp, predict_2d_kp):
        """
           purpose:
               calc L1 error
               \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
           Inputs:
               kp_gt  : N x K x 3
               kp_pred: N x K x 2
        """
        kp_gt = real_2d_kp.view(-1, 3).cuda()
        kp_pred = predict_2d_kp.contiguous().view(-1, 2)

        vis = kp_gt[:, 2]  # certainty of joint labels

        k = torch.sum(vis) * 2.0 + 1e-8
        diff_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)

        weighted_l1 = torch.matmul(diff_abs, vis)
        return weighted_l1 * 1.0 / k

    def batch_kp_3d_l2_loss(self, real_3d_kp, fake_3d_kp, w_3d):
        """
            purpose:
                calc mse * 0.5

            Inputs:
                real_3d_kp  : N x k x 3
                fake_3d_kp  : N x k x 3
                w_3d        : N x 1
        """
        shape = real_3d_kp.shape
        k = torch.sum(w_3d) * shape[1] * 3.0 * 2.0 + 1e-8

        # first align it
        real_3d_kp, fake_3d_kp = align_by_pelvis(real_3d_kp), align_by_pelvis(fake_3d_kp)
        kp_gt = real_3d_kp
        kp_pred = fake_3d_kp
        kp_dif = (kp_gt - kp_pred) ** 2
        return torch.matmul(kp_dif.sum(1).sum(1), w_3d) * 1.0 / k

    def batch_shape_l2_loss(self, real_shape, fake_shape, w_shape):
        """
           purpose:
               calc mse * 0.5

           Inputs:
               real_shape  :   N x 10
               fake_shape  :   N x 10
               w_shape     :   N x 1
        """
        k = torch.sum(w_shape) * 10.0 * 2.0 + 1e-8
        shape_dif = (real_shape - fake_shape) ** 2
        return torch.matmul(shape_dif.sum(1), w_shape) * 1.0 / k

    def batch_pose_l2_loss(self, real_pose, fake_pose, w_pose):
        """
            Input:
                real_pose   : N x 48
                fake_pose   : N x 48
        """

        k = torch.sum(w_pose) * 207.0 * 2.0 + 1e-8
        real_rs, fake_rs = batch_rodrigues(real_pose.view(-1, 3)).view(-1, 24, 9)[:, 1:, :], batch_rodrigues(
            fake_pose.view(-1, 3)).view(-1, 24, 9)[:, 1:, :]
        dif_rs = ((real_rs - fake_rs) ** 2).view(-1, 207)
        return torch.matmul(dif_rs.sum(1), w_pose) * 1.0 / k


def main():
    trainer = HMRTrainer()
    trainer.train()


if __name__ == '__main__':
    main()

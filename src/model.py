"""
    file:   model.py

    date:   2018_05_03
    author: zhangxiong(1025679612@qq.com)
"""

import sys

import torch
import torch.nn as nn
import numpy as np
import util
import pickle
import torchvision.models as models

from Discriminator import PoseDiscriminator, FullPoseDiscriminator
from MANO import MANO
from config import args
import config
import Resnet
from densenet import load_denseNet
from LinearModel import LinearModel


class ThetaRegressor(LinearModel):
    """
    Iterative Error Feedback loop
    """

    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func, iterations):
        super(ThetaRegressor, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)
        self.iterations = iterations
        batch_size = max(args.batch_size + args.batch_3d_size, args.eval_batch_size)
        mean_theta = np.tile(util.load_mean_theta(), batch_size).reshape((batch_size, -1))
        self.register_buffer('mean_theta', torch.from_numpy(mean_theta).float())

    def forward(self, inputs):
        """
            param:
                inputs: is the output of encoder, which has 2048 features

            return:
                a list contains [ [theta1, theta1, ..., theta1], [theta2, theta2, ..., theta2], ... , ], shape is iterations x N x 85(or other theta count)
        """
        thetas = []
        shape = inputs.shape
        theta = self.mean_theta[:shape[0], :]  # init theta with mean
        for _ in range(self.iterations):
            total_inputs = torch.cat([inputs, theta], 1)
            dTheta = self.fc_blocks(total_inputs)  # residual
            theta = theta + dTheta  # update theta with residuals from last run
            thetas.append(theta)
        return thetas


class HMRNetBase(nn.Module):
    def __init__(self):
        super(HMRNetBase, self).__init__()
        self._read_configs()

        print('start creating sub modules...')
        self._create_sub_modules()

    def _read_configs(self):
        def _check_config():
            encoder_name = args.encoder_network
            enable_inter_supervisions = args.enable_inter_supervision
            feature_count = args.feature_count
            if encoder_name == 'hourglass':
                assert args.crop_size == 256
            elif encoder_name == 'inception':
                assert args.crop_size == 299
            elif encoder_name == 'resnet50':
                assert args.crop_size == 224
                assert not enable_inter_supervisions
            elif encoder_name.startswith('densenet'):
                assert args.crop_size == 224
                assert not enable_inter_supervisions
            else:
                msg = 'invalid encoder network, only {} is allowed, got {}'.format(args.allowed_encoder_net,
                                                                                   encoder_name)
                sys.exit(msg)
            assert config.encoder_feature_count[encoder_name] == feature_count

        _check_config()

        self.encoder_name = args.encoder_network
        self.beta_count = args.beta_count

        with open(args.mano_model, 'rb') as f:
            self.mano_model = pickle.load(f, encoding='latin1')

        self.total_theta_count = args.total_theta_count
        self.joint_count = args.joint_count
        self.feature_count = args.feature_count  # number of neurons in dense layer

    def _create_sub_modules(self):
        """
            mano model, MANO can create a mesh from beta & theta
        """
        self.mano = MANO(self.mano_model, obj_saveable=True)

        '''
            Only resnet50 and hourglass is allowed currently, maybe other encoder will be allowed later.
        '''
        if self.encoder_name == 'resnet50':
            print('Creating resnet50')
            self.encoder = Resnet.load_Res50Model()

        elif self.encoder_name == 'inception':
            print('Creating inception')
            inception_model = models.inception_v3(pretrained=True)
            inception_layers = list(inception_model.children())
            del inception_layers[13]

            inception_layers[-1] = nn.AvgPool2d(35, stride=1)
            self.encoder = nn.Sequential(*inception_layers)

        elif self.encoder_name.startswith('densenet'):
            print('Creating densenet')
            self.encoder = load_denseNet(self.encoder_name)
        else:
            assert 0

        if not args.fine_tune:
            for param in self.encoder.parameters():  # disable training
                param.requires_grad = False

        '''
            Regressor can predict betas(include beta and theta which needed by MANO) from coder extracted from encoder in an iterative way
        '''
        fc_layers = [self.feature_count + self.total_theta_count, 1024, 1024, 61]
        use_dropout = [True, True, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False]  # deactivate the last layer
        iterations = 3
        self.regressor = ThetaRegressor(fc_layers, use_dropout, drop_prob, use_ac_func, iterations)
        self.iterations = iterations

        print('Finished creating the encoder modules...')

    def forward(self, inputs):
        if self.encoder_name == 'resnet50':
            feature = self.encoder(inputs)
            thetas = self.regressor(feature)
            detail_info = []
            for theta in thetas:
                detail_info.append(self._calc_detail_info(theta))
            return detail_info
        elif self.encoder_name.startswith('densenet'):
            feature = self.encoder(inputs)
            thetas = self.regressor(feature)
            detail_info = []
            for theta in thetas:
                detail_info.append(self._calc_detail_info(theta))
            return detail_info
        elif self.encoder_name.startswith('inception'):
            feature = self.encoder(inputs).view([-1, 2048])
            thetas = self.regressor(feature)
            detail_info = []
            for theta in thetas:
                detail_info.append(self._calc_detail_info(theta))
            return detail_info
        elif self.encoder_name == 'hourglass':
            if args.enable_inter_supervision:
                features = self.encoder(inputs)
                detail_info = []
                for feature in features:
                    thetas = self.regressor(feature)
                    detail_info.append(self._calc_detail_info(thetas[-1]))
                return detail_info
            else:
                features = self.encoder(inputs)
                thetas = self.regressor(features[-1])  # only the last block
                detail_info = []
                for theta in thetas:
                    detail_info.append(self._calc_detail_info(theta))
                return detail_info
        else:
            assert 0

    def _calc_detail_info(self, theta):
        """
            purpose:
                calc verts, joint2d, joint3d, Rotation matrix

            inputs:
                theta: N X (3 + 72 + 10)

            return:
                thetas, verts, j2d, j3d, Rs
        """
        cam = theta[:, 0:3].contiguous()
        pose = theta[:, 3:48 + 3].contiguous()
        shape = theta[:, 48 + 3:].contiguous()
        verts, j3d, Rs = self.mano(beta=shape, theta=pose, get_skin=True)
        j2d = util.batch_orth_proj(j3d, cam)

        return theta, verts, j2d, j3d, Rs


if __name__ == '__main__':
    cam = np.array([[0.9, 0, 0]], dtype=np.float)
    pose = np.array([[-9.44920200e+01, -4.25263865e+01, -1.30050643e+01, -2.79970490e-01,
                      3.24995661e-01, 5.03083125e-01, -6.90573755e-01, -4.12994214e-01,
                      -4.21870093e-01, 5.98717416e-01, -1.48420885e-02, -3.85911139e-02,
                      1.13642605e-01, 2.30647176e-01, -2.11843286e-01, 1.31767149e+00,
                      -6.61596447e-01, 4.02174644e-01, 3.03129424e-02, 5.91100770e-02,
                      -8.04416564e-02, -1.12944653e-01, 3.15045050e-01, -1.32838375e-01,
                      -1.33748209e-01, -4.99408923e-01, 1.40508643e-01, 6.10867911e-02,
                      -2.22951915e-02, -4.73448564e-02, -1.48489055e-01, 1.47620442e-01,
                      3.24157346e-01, 7.78414851e-04, 1.70687935e-01, -1.54716815e-01,
                      2.95053507e-01, -2.91967776e-01, 1.26000780e-01, 8.09572677e-02,
                      1.54710846e-02, -4.21941758e-01, 7.44124075e-02, 1.17146423e-01,
                      3.16305389e-01, 5.04810448e-01, -3.65526364e-01, 1.31366428e-01,
                      -2.76658949e-02, -9.17315987e-03, -1.88285742e-01, 7.86409877e-03,
                      -9.41106758e-02, 2.08424367e-01, 1.62278709e-01, -7.98170265e-01,
                      -3.97403587e-03, 1.11321421e-01, 6.07793270e-01, 1.42215980e-01,
                      4.48185010e-01, -1.38429048e-01, 3.77056061e-02, 4.48877661e-01,
                      1.31445158e-01, 5.07427503e-02, -3.80920772e-01, -2.52292254e-02,
                      -5.27745375e-02, -7.43903887e-02, 7.22498075e-02, -6.35824487e-03]])

    beta = np.array([[-3.54196257, 0.90870435, -1.0978663, -0.20436199, 0.18589762, 0.55789026, -0.18163599, 0.12002746,
                      -0.09172286, 0.4430783]])
    real_shapes = torch.from_numpy(beta).float().cuda()
    real_poses = torch.from_numpy(pose).float().cuda()

    net = HMRNetBase().cuda()
    nx = torch.rand(2, 3, 224, 224).float().cuda()

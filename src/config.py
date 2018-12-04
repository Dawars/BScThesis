"""
    file:   config.py

    date:   2018_04_29
    author: zhangxiong(1025679612@qq.com)
"""

import argparse

parser = argparse.ArgumentParser(description='hmr model')

parser.add_argument(
    '-f',
    default="",
    type=str,
    help='jupyter notebook workaround.'
)
parser.add_argument(
    '--fine-tune',
    default=True,
    type=bool,
    help='fine tune or not.'
)

parser.add_argument(
    '--encoder-network',
    type=str,
    default='resnet50',
    help='the encoder network name'
)

parser.add_argument(
    '--mano-model',
    type=str,
    default='/projects/pytorch_HMR/src/MANO_RIGHT_py3.pkl',
    help='mano model path'
)

parser.add_argument(
    '--total-theta-count',
    type=int,
    default=6 + 3 * 15 + 10,  # camera(scale, translation^2, rotation^3) + poses(45) + shapes(10)
    help='the count of theta param'
)

parser.add_argument(
    '--batch-size',
    type=int,
    default=64,
    help='batch size'
)

parser.add_argument(
    '--batch-3d-size',
    type=int,
    default=0,
    help='3d data batch size'
)

parser.add_argument(
    '--adv-batch-size',
    type=int,
    default=128,
    help='default adv batch size'
)

parser.add_argument(
    '--eval-batch-size',
    type=int,
    default=32,
    help='default eval batch size'
)

parser.add_argument(
    '--joint-count',
    type=int,
    default=16,
    help='the count of joints'
)

parser.add_argument(
    '--beta-count',
    type=int,
    default=10,
    help='the count of beta'
)

parser.add_argument(
    '--use-adv-train',
    type=bool,
    default=False,
    help='use adv traing or not'
)

parser.add_argument(
    '--scale-min',
    type=float,
    default=1.1,
    help='min scale'
)

parser.add_argument(
    '--scale-max',
    type=float,
    default=1.5,
    help='max scale'
)

parser.add_argument(
    '--num-worker',
    type=int,
    default=0,
    help='pytorch number worker.'
)

parser.add_argument(
    '--iter-count',
    type=int,
    default=100001,
    help='iter count, each contains batch-size samples'
)

parser.add_argument(
    '--e-lr',
    type=float,
    default=0.00001,
    help='encoder learning rate.'
)

parser.add_argument(
    '--d-lr',
    type=float,
    default=0.0001,
    help='Adversarial prior learning rate.'
)

parser.add_argument(
    '--e-wd',
    type=float,
    default=0.0001,
    help='encoder weight decay rate.'
)

parser.add_argument(
    '--d-wd',
    type=float,
    default=0.0001,
    help='Adversarial prior weight decay'
)

parser.add_argument(
    '--e-loss-weight',
    type=float,
    default=100,
    help='weight on encoder 2d kp losses.'
)

parser.add_argument(
    '--d-loss-weight',
    type=float,
    default=1,
    help='weight on discriminator losses'
)

parser.add_argument(
    '--d-disc-ratio',
    type=float,
    default=1.0,
    help='multiple weight of discriminator loss'
)

parser.add_argument(
    '--e-3d-loss-weight',
    type=float,
    default=60,
    help='weight on encoder thetas losses.'
)

parser.add_argument(
    '--e-shape-ratio',
    type=float,
    default=5,
    help='multiple weight of shape loss'
)

parser.add_argument(
    '--e-3d-kp-ratio',
    type=float,
    default=10.0,
    help='multiple weight of 3d key point.'
)

parser.add_argument(
    '--e-pose-ratio',
    type=float,
    default=20,
    help='multiple weight of pose'
)

parser.add_argument(
    '--save-folder',
    type=str,
    default='/mnt/dawars/hdd1/model_saves/hand_10/',
    help='save model path'
)

parser.add_argument(  # look into intermediate layers
    '--enable-inter-supervision',
    type=bool,
    default=False,
    help='enable inter supervision or not.'
)

train_2d_set = ['11k_train']
train_3d_set = []
train_adv_set = ['mano']
eval_set = ['11k_val']

allowed_encoder_net = ['hourglass', 'resnet50', 'inception', 'densenet169', 'densenet121']

encoder_feature_count = {
    'hourglass': 4096,
    'resnet50': 2048,
    'inception': 2048,
    'densenet169': 1664,
    'densenet121': 1024
}

crop_size = {
    'hourglass': 256,
    'resnet50': 224,
    'inception': 299,
    'densenet169': 224,
    'densenet121': 224,
}

data_set_path = {
    '11k_train': '/projects/pytorch_HMR/src/11k_joints_train.pkl',
    '11k_val': '/projects/pytorch_HMR/src/11k_joints_val.pkl',
    'mano': '/projects/pytorch_HMR/src/MANO_RIGHT_py3.pkl',
}

pre_trained_model = {
    'generator': '/HMR/hmr_resnet50/fine_tuned/3500_generator.pkl',
    'discriminator': '/HMR/hmr_resnet50/fine_tuned/3500_discriminator.pkl'
}

args = parser.parse_args()
encoder_network = args.encoder_network
args.feature_count = encoder_feature_count[encoder_network]
args.crop_size = crop_size[encoder_network]

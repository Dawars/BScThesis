import pickle
import torch

import numpy as np

from torch.utils.data import Dataset


class ManoPoseDataloader(Dataset):
    def __init__(self, data_set_path):
        self.data_path = data_set_path

        self._load_data_set()

    def _load_data_set(self):
        with open(self.data_path, 'rb') as f:
            mano_data = pickle.load(f, encoding='latin1')

            hands_components = mano_data['hands_components']
            hands_coeffs = mano_data['hands_coeffs']
            hands_pose_mean = mano_data['hands_mean']

            # 3*15 = 45 joint angles
            self.poses = np.matmul(hands_coeffs, hands_components) + hands_pose_mean

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        trival, pose, shape = np.zeros(3), self.poses[index], np.zeros([10])

        return {
            'theta': torch.tensor(np.concatenate((trival, pose, shape), axis=0)).float()
        }


if __name__ == '__main__':
    mano = ManoPoseDataloader('/projects/pytorch_HMR/src/MANO_RIGHT_py3.pkl')

    for sample in mano:
        print(sample)

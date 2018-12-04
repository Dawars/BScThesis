import sys
from config import args
import pickle

import numpy as np
import torch
import torch.nn as nn

from util import batch_global_rigid_transformation, batch_rodrigues, batch_lrotmin, reflect_pose, undo_chumpy


class MANO(nn.Module):
    def __init__(self, dd, obj_saveable=False, dtype=torch.float):
        super(MANO, self).__init__()

        self.register_buffer('v_template', torch.tensor(
            undo_chumpy(dd['v_template']),
            dtype=dtype,
            requires_grad=False))
        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]

        if obj_saveable:
            self.faces = dd['f']
        else:
            self.faces = None

        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis: 778 x 3 x 10
        # reshaped to 778*30 x 10, transposed to 10x778*3
        shapedir = np.reshape(undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.tensor(shapedir, dtype=dtype, requires_grad=False))

        # Regressor for joint locations given shape - 778 x 16
        self.register_buffer('J_regressor',
                             torch.tensor(dd['J_regressor'].T.todense(),
                                          dtype=dtype,
                                          requires_grad=False))

        # Pose blend shape basis: 778 x 3 x 135, reshaped to 778*3 x 135
        num_pose_basis = dd['posedirs'].shape[-1]
        # 135 x 2334
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.tensor(
            posedirs, dtype=dtype, requires_grad=False))
        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self.register_buffer('weights', torch.tensor(
            undo_chumpy(dd['weights']),
            dtype=dtype,
            requires_grad=False))

        # This returns 15 keypoints: 778 x 16
        self.register_buffer('joint_regressor', torch.tensor(
            dd['J_regressor'].T.todense(),
            dtype=dtype,
            requires_grad=False))

        self.register_buffer('e3', torch.eye(3).float())

        self.cur_device = None

    def save_obj(self, verts, obj_mesh_name):
        if self.faces is None:
            msg = 'obj not saveable!'
            sys.exit(msg)

        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces + 1:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def forward(self, beta, theta, get_skin=False):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
            beta: N x 10
            theta: N x 48 (with 3-D axis-angle rep) [float]
        Updates:
        self.J_transformed: N x 16 x 3 joint location after shaping
                & posing with beta and theta
        Returns:
         - joints: N x 16 joint locations depending on joint_type
        If get_skin is True, also returns
         - Verts: N x 778 x 3
       """
        if not self.cur_device:
            device = beta.device
            self.cur_device = torch.device(device.type, device.index)

        num_batch = beta.shape[0]

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 16, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 135)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base=True)

        W = self.weights.repeat(num_batch, 1).view(num_batch, -1, 16)
        T = torch.matmul(W, A.view(num_batch, 16, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=self.cur_device)], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]

        joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor)

        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints


if __name__ == '__main__':
    with open('/projects/pytorch_HMR/src/MANO_RIGHT_py3.pkl', 'rb') as f:
        mano_data = pickle.load(f, encoding='latin1')

    device = torch.device('cuda', 0)

    mano = MANO(mano_data, obj_saveable=True).to(device)
    pose = np.zeros([48])
    pose[3:] = np.array([0.0423, -2.1048, -1.4226, -1.1823, 5.1067, 2.3390, -1.2804, -2.1466,
                         -1.7817, 0.3036, -1.6508, 2.0646, 0.9613, 0.3191, 0.9910, -0.9373,
                         -2.6143, -2.0367, -1.4767, -0.6670, -1.5367, 1.3100, -0.7765, 3.9728,
                         2.1709, -1.8414, 0.4141, 0.8715, -0.4011, 3.0651, 0.4543, 1.6731,
                         -0.0783, 0.0948, 0.3612, -2.1473, 0.7537, 0.6456, 0.5057, -2.1914,
                         -2.2966, -0.0081, 0.3713, 1.3000, -0.5734])
    # beta = np.array([-1.0558, 3.6000, 4.0335, -1.9006, 3.6700, -0.0602, 4.8482, 5.2511,-1.8204, -3.8448])
    beta = np.zeros([10])
    vbeta = torch.tensor(np.array([beta])).float().to(device)
    vpose = torch.tensor(np.array([pose])).float().to(device)

    verts, j, r = mano(vbeta, vpose, get_skin=True)

    mano.save_obj(verts[0].cpu().numpy(), './mesh.obj')

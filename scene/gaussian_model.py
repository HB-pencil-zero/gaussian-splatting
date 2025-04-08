#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args , only_shs = False, wo_xyz = False):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        if only_shs:
            lr_scale = 1
            l = [
                {'params': [self._features_dc], 'lr': training_args.feature_lr * lr_scale, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20 * lr_scale, "name": "f_rest"},
                # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            ]
        elif wo_xyz:
            l = [
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]
        else :
            lr_scale = 1
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale * lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr * lr_scale, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0 * lr_scale, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr * lr_scale, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr * lr_scale, "name": "rotation"}
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    from plyfile import PlyData

    def check_attribute_shapes(self, ply_file):
        """
        检查 .ply 文件中所有属性的值是否与 `x` 的形状一致。

        参数:
            ply_file (str): .ply 文件路径。

        返回:
            dict: 每个属性及其与 `x` 的形状是否一致。
        """
        # 加载 .ply 文件
        plydata = PlyData.read(ply_file)
        
        # 获取 `x` 的形状
        x_shape = plydata['vertex']['x'].shape

        # 存储属性和结果
        results = {}

        # 遍历所有属性，检查形状
        for attribute in plydata['vertex'].data.dtype.names:
            attr_shape = plydata['vertex'][attribute].shape
            results[attribute] = (attr_shape == x_shape)

        return results


    
    
    def load_ply(self, path, default_filling=True):
        plydata = PlyData.read(path)
        
        
        
        # flag_path = os.path.join(os.path.dirname(path), 'filled.pt')
    
        # # 检查是否存在标志文件
        # if os.path.exists(flag_path):
        #     self.filled = torch.load(flag_path)
        # else:
        #     # 默认为True并保存
        #     self.filled = torch.tensor(default_filling)
        #     torch.save(self.filled, flag_path)


        # plydata2 = filter_ply_entries_by_hyperplane(plydata)
        # plydata3 = filter_ply_entries_by_hyperplane(plydata2, b=0.1, pos=False)
        # plydata = plydata3
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
    
    def fill_interior(self, voxel_resolution=128, density_threshold=0.1, scaling_factor=1/3000, opacity_value=0.99):
        """
        填充3D对象内部的空白区域，实现FruitNinja中描述的OpaqueAtom GS策略填充方法。
        
        参数:
            voxel_resolution (int): 3D体素网格的分辨率
            density_threshold (float): 判断区域为空的密度阈值
            scaling_factor (float): 高斯粒子大小的缩放因子 (相对于场景尺寸的比例)
            opacity_value (float): 填充粒子的不透明度值 (接近1表示完全不透明)
        """
        import torch
        import numpy as np
        from torch.nn.functional import conv3d
        from simple_knn._C import distCUDA2  # 使用已有的distCUDA2替代knnsearch
        
        # 计算场景边界
        xyz = self.get_xyz.detach()
        min_coords = torch.min(xyz, dim=0).values
        max_coords = torch.max(xyz, dim=0).values
        scene_extent = torch.max(max_coords - min_coords)
        
        # 创建体素网格
        print(f"创建3D体素网格: {voxel_resolution}x{voxel_resolution}x{voxel_resolution}")
        voxel_size = scene_extent / voxel_resolution
        grid_min = min_coords - voxel_size
        grid_max = max_coords + voxel_size
        
        # 初始化密度网格
        density_grid = torch.zeros((voxel_resolution, voxel_resolution, voxel_resolution), 
                                device=xyz.device, dtype=torch.float)
        
        # 为体素网格创建中心点坐标
        steps = torch.linspace(0, voxel_resolution-1, voxel_resolution, device=xyz.device)
        grid_x = grid_min[0] + steps * (grid_max[0] - grid_min[0]) / (voxel_resolution - 1)
        grid_y = grid_min[1] + steps * (grid_max[1] - grid_min[1]) / (voxel_resolution - 1)
        grid_z = grid_min[2] + steps * (grid_max[2] - grid_min[2]) / (voxel_resolution - 1)
        
        # 计算每个粒子对密度网格的贡献
        scaling = self.get_scaling.detach()
        opacity = self.get_opacity.detach()
        
        # 对每个体素进行密度采样
        print("计算密度网格...")
        
        # 采样点限制，避免内存溢出
        sample_limit = 10000
        num_particles = xyz.shape[0]
        
        for i in range(0, num_particles, sample_limit):
            end_idx = min(i + sample_limit, num_particles)
            batch_xyz = xyz[i:end_idx]
            batch_opacity = opacity[i:end_idx]
            
            # 计算体素索引
            voxel_idx_x = ((batch_xyz[:, 0] - grid_min[0]) / (grid_max[0] - grid_min[0]) * (voxel_resolution - 1)).long()
            voxel_idx_y = ((batch_xyz[:, 1] - grid_min[1]) / (grid_max[1] - grid_min[1]) * (voxel_resolution - 1)).long()
            voxel_idx_z = ((batch_xyz[:, 2] - grid_min[2]) / (grid_max[2] - grid_min[2]) * (voxel_resolution - 1)).long()
            
            # 确保索引在有效范围内
            mask = (voxel_idx_x >= 0) & (voxel_idx_x < voxel_resolution) & \
                (voxel_idx_y >= 0) & (voxel_idx_y < voxel_resolution) & \
                (voxel_idx_z >= 0) & (voxel_idx_z < voxel_resolution)
            
            voxel_idx_x = voxel_idx_x[mask]
            voxel_idx_y = voxel_idx_y[mask]
            voxel_idx_z = voxel_idx_z[mask]
            batch_opacity = batch_opacity[mask]
            
            # 更新密度网格
            density_grid[voxel_idx_x, voxel_idx_y, voxel_idx_z] += batch_opacity.squeeze()
        
        # 识别内部空白区域
        filled_mask = density_grid < density_threshold
        
        # 通过卷积操作查找内部区域
        kernel_size = 5
        kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=xyz.device)
        input_tensor = density_grid.reshape(1, 1, voxel_resolution, voxel_resolution, voxel_resolution)
        
        print("识别内部空白区域...")
        with torch.no_grad():
            neighbor_sum = conv3d(input_tensor, kernel, padding=kernel_size//2)
        neighbor_sum = neighbor_sum.reshape(voxel_resolution, voxel_resolution, voxel_resolution)
        
        # 如果周围有高密度，则为内部体素
        interior_voxels = (filled_mask) & (neighbor_sum > 1.0)
        
        # 选择要填充的体素坐标
        interior_indices = torch.nonzero(interior_voxels)
        
        if len(interior_indices) == 0:
            print("未找到需要填充的内部区域")
            return
        
        print(f"找到 {len(interior_indices)} 个内部体素需要填充")
        
        # 从内部体素生成新粒子
        from tqdm import tqdm
        
        new_xyz_list = []
        for idx in tqdm(interior_indices, desc="生成新粒子坐标"):
            x = grid_min[0] + idx[0].item() * (grid_max[0] - grid_min[0]) / (voxel_resolution - 1)
            y = grid_min[1] + idx[1].item() * (grid_max[1] - grid_min[1]) / (voxel_resolution - 1)
            z = grid_min[2] + idx[2].item() * (grid_max[2] - grid_min[2]) / (voxel_resolution - 1)
            
            # 添加一些随机扰动以避免规则格子状排列
            jitter = (torch.rand(3, device=xyz.device) - 0.5) * voxel_size * 0.5
            new_xyz_list.append([x + jitter[0], y + jitter[1], z + jitter[2]])
        
        # 限制填充的粒子数量，避免内存溢出
        max_fill_particles = 20000000
        if len(new_xyz_list) > max_fill_particles:
            print(f"限制填充粒子数量为 {max_fill_particles}")
            indices = torch.randperm(len(new_xyz_list))[:max_fill_particles]
            new_xyz_list = [new_xyz_list[idx.item()] for idx in indices]
        
        # 创建新的粒子属性
        new_xyz = torch.tensor(new_xyz_list, dtype=torch.float, device=xyz.device)
        num_new_particles = len(new_xyz)
        
        # 应用OpaqueAtom GS设置
        # 1. 原子裁剪：限制高斯粒子的大小
        atomic_scale = scene_extent * scaling_factor
        new_scaling = torch.ones((num_new_particles, 3), device=xyz.device) * atomic_scale
        new_scaling = self.scaling_inverse_activation(new_scaling)
        
        # 2. 均匀不透明化：分配完全不透明度
        new_opacity = torch.ones((num_new_particles, 1), device=xyz.device) * opacity_value
        new_opacity = self.inverse_opacity_activation(new_opacity)
        
        # 设置旋转属性为单位四元数
        new_rotation = torch.zeros((num_new_particles, 4), device=xyz.device)
        new_rotation[:, 0] = 1.0  # 单位四元数 [1, 0, 0, 0]
        
        # 设置颜色特征
        new_features_dc = torch.zeros((num_new_particles, 1, 3), device=xyz.device)
        new_features_rest = torch.zeros((num_new_particles, (self.max_sh_degree + 1) ** 2 - 1, 3), device=xyz.device)
        
        # 为每个新粒子找到最近的表面粒子并采用其颜色
        print("为内部粒子分配颜色...")
        batch_size = min(10000, num_new_particles)
        
        # 使用批处理方式计算最近邻
        for i in range(0, num_new_particles, batch_size):
            end_idx = min(i + batch_size, num_new_particles)
            batch_new_xyz = new_xyz[i:end_idx]
            
            # 查找K个最近邻 - 使用distCUDA2计算距离
            k = 5
            # 对每个新点计算与所有已有点的距离
            batch_dists = []
            batch_idxs = []
            
            for j in range(batch_new_xyz.shape[0]):
                point = batch_new_xyz[j:j+1]
                # 使用distCUDA2计算点到所有已存在点的距离
                dists = distCUDA2(point, xyz)
                # 找到最近的k个点
                topk_dists, topk_idx = torch.topk(dists, k=k, largest=False)
                batch_dists.append(topk_dists)
                batch_idxs.append(topk_idx)
                
            # 将列表转换为张量
            dists = torch.stack(batch_dists)
            idxs = torch.stack(batch_idxs)
            
            # 基于距离加权平均
            weights = torch.exp(-dists)
            weights_sum = weights.sum(dim=1, keepdim=True)
            weights = weights / (weights_sum + 1e-8)
            
            # 获取并应用加权颜色
            for j in range(k):
                neighbor_indices = idxs[:, j]
                neighbor_dc = self._features_dc[neighbor_indices]
                neighbor_rest = self._features_rest[neighbor_indices]
                
                weight_factor = weights[:, j].view(-1, 1, 1)
                new_features_dc[i:end_idx] += neighbor_dc * weight_factor
                new_features_rest[i:end_idx] += neighbor_rest * weight_factor
        
        print(f"向模型添加 {num_new_particles} 个填充粒子...")
        
        # 添加新粒子到模型
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        
        # 保存填充标志，表示模型已经被填充
        self.filled = torch.tensor(True, device=xyz.device)
        print("内部填充完成")
                        

import torch

import pyelsed
import cv2
from functions.voxel_helpers import ray_intersect
from utils.filter_utils import edge_conv2d, dist_conv2d
from utils.sample_util import updownsampling_voxel

def dir2tilt_rot(dir):
    # get the tilt matrix
    rot_matrix = torch.zeros(
        (dir.shape[0], dir.shape[1], 3, 3)).cuda()
    bx = dir[:, :, 0]
    by = dir[:, :, 1]
    bz = dir[:, :, 2]
    rot_matrix[:, :, 0, 0] = bz + (by ** 2) / (1 + bz)
    rot_matrix[:, :, 0, 1] = -bx * by / (1 + bz)
    rot_matrix[:, :, 0, 2] = bx
    rot_matrix[:, :, 1, 0] = -bx * by / (1 + bz)
    rot_matrix[:, :, 1, 1] = 1 - by ** 2 / (1 + bz)
    rot_matrix[:, :, 1, 2] = by
    rot_matrix[:, :, 2, 0] = -bx
    rot_matrix[:, :, 2, 1] = -by
    rot_matrix[:, :, 2, 2] = bz
    # note:
    rot_matrix = rot_matrix.transpose(-1, -2)
    return rot_matrix

class low_freq_directions():
    def __init__(self, line_similar_th = 0.95, pc_dist_th=0.05, ksize=0, valid_line_th = 1e-2, max_low_freq_dir_num=2, num_vertexes=20000):
        self.compress_dir_info_voxel = torch.zeros(
            (num_vertexes, max_low_freq_dir_num, 3),
            requires_grad=False, dtype=torch.float32,
            device=torch.device("cuda"))
        self.compress_rot_info_voxel = torch.nan * torch.ones(
            (num_vertexes, max_low_freq_dir_num, 3, 3),
            requires_grad=False, dtype=torch.float32,
            device=torch.device("cuda"))
        self.compress_weight_info_voxel = torch.zeros(
            (num_vertexes, max_low_freq_dir_num, 1),
            requires_grad=False, dtype=torch.float32,
            device=torch.device("cuda"))
        self.dist_conv = dist_conv2d(3, "cuda")
        self.color_conv = edge_conv2d(3, "cuda")
        self.depth_conv = edge_conv2d(1, "cuda")
        self.line_similar_th = line_similar_th
        self.pc_dist_th = pc_dist_th
        self.ksize = ksize
        self.valid_line_th = valid_line_th
        self.max_dir_num = max_low_freq_dir_num

    def texture_pattern_determination(self, img_raw, pc_img, map_states, voxel_size, w2c, K):
        highest_score_segments_dir_all, highest_score_segments_weight_all = self.fuse_line_segment_directions_per_frame(img_raw, pc_img, map_states, voxel_size, w2c, K)
        self.update_low_frequency_directions_across_frames(highest_score_segments_dir_all, highest_score_segments_weight_all)

    # Fuse Line Segment Directions in each Frame
    def fuse_line_segment_directions_per_frame(self, img_raw, pc_img, map_states, voxel_size, w2c, K):
        self.line_segments(img_raw, pc_img, w2c, K) # get the valid 3D line segments
        self.get_line_voxel_intersect(map_states, voxel_size) # allocate each line to the cooresponding voxel
        if self.hits.sum() > 0:
            self.get_group_info() # sort the line segments into each voxel
            highest_score_segments_dir_all, highest_score_segments_weight_all = [],[]
            for i in range(self.max_dir_num):
                # iteratively select the line segments that have the highest score and remove the similiar ones
                weighted_highest_score_segments_dir, highest_score_segments_weight = self.select_dir(self.line_similar_th)
                highest_score_segments_dir_all.append(weighted_highest_score_segments_dir.unsqueeze(1))
                highest_score_segments_weight_all.append(highest_score_segments_weight.unsqueeze(1))
            highest_score_segments_dir_all = torch.concatenate(highest_score_segments_dir_all, dim=1)
            highest_score_segments_weight_all = torch.concatenate(highest_score_segments_weight_all, dim=1)
            return highest_score_segments_dir_all, highest_score_segments_weight_all
        else:
            return None, None

    def update_low_frequency_directions_across_frames(self, highest_score_segments_dir_all, highest_score_segments_weight_all):
        '''
        # insert unsimilar dir into empty space
        # set the similar directions to zero, then place the unsimilar directions at the front,
        and also set the indices of the empty spaces to true and place them at the front.
        This ensures that the two indices are consistent, allowing us to retrieve the unsimilar directions based on the number of empty spaces.
        (If the number of empty spaces is greater than the number of unsimilar directions, the directions set to zero will be retrieved
        '''
        origin_compress_dir_info_voxel = self.compress_dir_info_voxel[self.unique_intersect_voxel_idx].clone()
        origin_compress_weight_info_voxel = self.compress_weight_info_voxel[self.unique_intersect_voxel_idx].clone()
        batch_idx = torch.arange(origin_compress_dir_info_voxel.shape[0]).cuda()
        cos_dir_origin_new_tmp = (
                origin_compress_dir_info_voxel.unsqueeze(1) * highest_score_segments_dir_all.unsqueeze(2)).sum(dim=-1)
        cos_dir_origin_new = cos_dir_origin_new_tmp.abs()
        # whether the selected dir has similar dir in the origin dir
        unsimilar_all_origin_mask = ((cos_dir_origin_new > self.line_similar_th) == False).all(dim=1)
        empty_dir_mask = (origin_compress_dir_info_voxel == 0).all(dim=-1)

        # make the true in front of false
        sorted_empty_dir_mask, _ = torch.sort(empty_dir_mask.int(), dim=-1, descending=True)
        sorted_empty_dir_mask = sorted_empty_dir_mask.bool()
        unsimilar_all_origin_mask_tmp = unsimilar_all_origin_mask.clone()
        # not consider the invalid dir
        unsimilar_all_origin_mask_tmp[(highest_score_segments_dir_all == 0).all(dim=-1)] = False
        # make the true in front of false
        sorted_unsimilar_idx = unsimilar_all_origin_mask_tmp.int().argsort(dim=-1, descending=True, stable=True)
        # change the unsimilar dir in front of the similar dir
        sorted_longest_segments_dir_all = highest_score_segments_dir_all.clone()
        # if there is an empty space, but all the dir are similar to the origin dir, then this dir should not be inserted
        sorted_longest_segments_dir_all[~unsimilar_all_origin_mask] = 0
        sorted_longest_segments_dir_all = sorted_longest_segments_dir_all[
            torch.arange(sorted_longest_segments_dir_all.shape[0]).cuda().unsqueeze(-1), sorted_unsimilar_idx]
        # sorted weight as dir
        sorted_segments_weight_all = highest_score_segments_weight_all.clone()
        sorted_segments_weight_all[~unsimilar_all_origin_mask] = 0
        sorted_segments_weight_all = sorted_segments_weight_all[batch_idx.unsqueeze(-1), sorted_unsimilar_idx]
        # insert the unsimilar dir to the empty space
        new_compress_dir_info_voxel = origin_compress_dir_info_voxel.clone()
        new_compress_dir_info_voxel[empty_dir_mask] = sorted_longest_segments_dir_all[sorted_empty_dir_mask]
        new_compress_weight_info_voxel = origin_compress_weight_info_voxel.clone()
        new_compress_weight_info_voxel[empty_dir_mask] = sorted_segments_weight_all[sorted_empty_dir_mask]
        '''
        # fuse the similar dir to the origin dir
        '''
        # find which new dir is similar to the original dir, note: the idx is th id of new
        similar_longest_dir_idx = cos_dir_origin_new.max(dim=2)[1]
        # get the mask of whether the new dir is similar to the origin dir, note: the mask is of the new dir
        similar_origin_dir_mask = ~unsimilar_all_origin_mask
        similar_one_origin_mask_tmp = similar_origin_dir_mask.clone()
        # not consider the invalid dir
        similar_one_origin_mask_tmp[(highest_score_segments_dir_all == 0).all(dim=-1)] = False
        # get the weighted dir that should be fused to the origin dir
        similar_longest_weight = highest_score_segments_weight_all[batch_idx.unsqueeze(-1), similar_longest_dir_idx]
        similar_longest_dir = highest_score_segments_dir_all[batch_idx.unsqueeze(-1), similar_longest_dir_idx]
        similar_longest_weight[~similar_one_origin_mask_tmp] = 0
        similar_weighted_longest_dir = similar_longest_weight * similar_longest_dir
        # if the origin dir is similar to the two new dir, combine the info of the two new dir
        similar_longest_weight = updownsampling_voxel(similar_longest_weight, similar_longest_dir_idx,
                                                      torch.zeros(2).cuda(), mode="sum")
        similar_weighted_longest_dir = updownsampling_voxel(similar_weighted_longest_dir, similar_longest_dir_idx,
                                                            torch.zeros(2).cuda(), mode="sum")
        # insert the fused dir to the non empty space
        original_weighted_dir = origin_compress_weight_info_voxel * origin_compress_dir_info_voxel
        new_compress_dir_info_voxel[~empty_dir_mask] = (similar_weighted_longest_dir[~empty_dir_mask] +
                                                        original_weighted_dir[~empty_dir_mask]) / \
                                                       (similar_longest_weight[~empty_dir_mask] +
                                                        origin_compress_weight_info_voxel[
                                                            ~empty_dir_mask])
        new_compress_dir_info_voxel[empty_dir_mask] /= new_compress_dir_info_voxel[empty_dir_mask].norm(dim=-1,
                                                                                                        keepdim=True)
        new_compress_weight_info_voxel[~empty_dir_mask] += similar_longest_weight[~empty_dir_mask]

        self.compress_dir_info_voxel[self.unique_intersect_voxel_idx] = new_compress_dir_info_voxel
        self.compress_weight_info_voxel[self.unique_intersect_voxel_idx] = new_compress_weight_info_voxel

        new_compress_dir_info_voxel[(new_compress_dir_info_voxel == 0).all(-1)] = torch.nan # nan represents not this texture pattern
        # get the tilt matrix
        rot_matrix = dir2tilt_rot(new_compress_dir_info_voxel)
        self.compress_rot_info_voxel[self.unique_intersect_voxel_idx] = rot_matrix

    def line_segments(self, img_raw, pc_img, w2c, K):
        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        dist = self.dist_conv(pc_img.permute(2, 0, 1).unsqueeze(0))
        max_dist = dist.abs().max(dim=1)[0].squeeze()
        segments, scores = pyelsed.detect_consider_pc_dist(gray.copy(), max_dist.detach().cpu().numpy().copy(),
                                                           pc_dist_threshold=self.pc_dist_th, ksize=self.ksize)
        segments = torch.from_numpy(segments).cuda() # detected segments
        scores = torch.from_numpy(scores).cuda() # the rate of inner point when fit a line
        segments = segments.clamp(min=0, max=img_raw.shape[1] - 1)

        # get the valid segments that is a line in 3d
        self.valid_segments_begin, self.valid_segments_dir, self.valid_segments_length, self.valid_scores = \
            self.line_validation(segments, scores, img_raw, pc_img, w2c, K)

    # sample in 3d then project to 2d, then back project to 3d to validate line
    def line_validation(self, segments, scores, img_raw, pc_img, w2c, K):
        segments_begin_end = segments.reshape(-1, 2, 2).clone()
        # turn to [-1,1]
        segments_begin_end[:, :, 0] = 2 * (segments_begin_end[:, :, 0] - img_raw.shape[1] / 2) / img_raw.shape[1]
        segments_begin_end[:, :, 1] = 2 * (segments_begin_end[:, :, 1] - img_raw.shape[0] / 2) / img_raw.shape[0]
        # segments coordinate is (x,y), match the grid_sample
        segments_sample_pc_begin_end = torch.nn.functional.grid_sample(pc_img.permute(2, 0, 1).unsqueeze(0),
                                                                       segments_begin_end.unsqueeze(0),
                                                                       padding_mode='border',
                                                                       mode='nearest', align_corners=False)
        segments_sample_pc_begin_end = segments_sample_pc_begin_end.permute(0, 2, 3, 1).squeeze(0)
        # sample numbers to validate line
        num_samples = 5
        # generate the sample points in 2D
        weights = torch.linspace(0, 1, num_samples).reshape(1, num_samples, 1).cuda()
        segments_sample_pc = segments_sample_pc_begin_end[:, 0:1, :] + weights * (
                segments_sample_pc_begin_end[:, 1:2, :] - segments_sample_pc_begin_end[:, 0:1, :])

        segments_sample_pc_tmp = segments_sample_pc.reshape(-1, 3)
        w2c = torch.linalg.inv(w2c.float())
        ones = torch.ones_like(segments_sample_pc_tmp[:, 0]).reshape(-1, 1).float()
        homo_points = torch.cat([segments_sample_pc_tmp, ones], dim=-1).unsqueeze(-1).float()
        homo_cam_points = w2c @ homo_points  # (N,4,1) = (4,4) * (N,4,1)
        cam_points = homo_cam_points[:, :3]  # (N,3,1)
        uv = K.float() @ cam_points.float()
        z = uv[:, -1:] + 1e-8
        uv = uv[:, :2] / z  # (N,2)
        uv = uv.round()  # (N_mask,1) -> (N_mask)
        # this dim of uv is different from frame.rgb, same as segment
        uv = uv.reshape(-1, num_samples, 2)
        uv[:, :, 0] = 2 * (uv[:, :, 0] - img_raw.shape[1] / 2) / img_raw.shape[1]
        uv[:, :, 1] = 2 * (uv[:, :, 1] - img_raw.shape[0] / 2) / img_raw.shape[0]
        # uv coordinate is (x,y), match the grid_sample
        uv_pc = torch.nn.functional.grid_sample(pc_img.permute(2, 0, 1).unsqueeze(0),
                                                uv.unsqueeze(0), padding_mode='border',
                                                mode='nearest', align_corners=False)
        uv_pc = uv_pc.permute(0, 2, 3, 1).squeeze(0)
        # validate line
        error = (uv_pc - segments_sample_pc).abs().sum(dim=1).sum(dim=1)
        valid_line_mask = error < self.valid_line_th

        valid_scores = scores[valid_line_mask]
        valid_segments_begin = segments_sample_pc[valid_line_mask, 0]
        valid_segments_end = segments_sample_pc[valid_line_mask, -1]
        valid_segments_dir = valid_segments_end - valid_segments_begin
        valid_segments_dir = valid_segments_dir / valid_segments_dir.norm(dim=1).unsqueeze(1)
        valid_segments_length = (valid_segments_end - valid_segments_begin).norm(dim=1)

        return valid_segments_begin, valid_segments_dir, valid_segments_length, valid_scores

    def get_line_voxel_intersect(self, map_states, voxel_size):
        centres = map_states["voxel_center_xyz"]
        childrens = map_states["voxel_structure"]

        intersections, hits = ray_intersect(
            self.valid_segments_begin.unsqueeze(0), self.valid_segments_dir.unsqueeze(0), centres,
            childrens, voxel_size, jcx_max_hits=100, max_distance=100)
        self.ray_mask = hits.view(1, -1)
        self.intersect_voxel_idx = intersections['intersected_voxel_idx'][0]
        self.intersect_min_depth = intersections['min_depth'][0]
        self.hits = hits

    def get_group_info(self):
        valid_intersect_mask = self.intersect_min_depth <= self.valid_segments_length.unsqueeze(1)

        # choose value based on ray_mask
        valid_intersect_mask = valid_intersect_mask[self.ray_mask[0]]
        valid_segments_dir = self.valid_segments_dir[self.ray_mask[0]]
        intersect_voxel_idx = self.intersect_voxel_idx[self.ray_mask[0]]
        valid_scores = self.valid_scores[self.ray_mask[0]]

        # choose value from valid_segments_dir based on valid_intersect_mask, the masked dir is all zero
        masked_valid_segments_dir = valid_segments_dir.unsqueeze(1) * valid_intersect_mask.unsqueeze(-1)
        masked_valid_segments_dir = masked_valid_segments_dir[(masked_valid_segments_dir != 0).any(dim=-1)]
        masked_intersect_voxel_idx = intersect_voxel_idx[valid_intersect_mask]
        masked_valid_scores = valid_scores.unsqueeze(-1).repeat(1, valid_intersect_mask.shape[1])[valid_intersect_mask]
        unique_intersect_voxel_idx, intersect_inverse_indices, intersect_counts = torch.unique(
            masked_intersect_voxel_idx,
            dim=0, return_inverse=True, return_counts=True)

        # get the segment dir group, (m,n,3), m is the number of intersected voxels, n is the number of segments
        # note: m is sorted by the order of unique_intersect_voxel_idx
        segments_dir_group_mask = torch.arange(intersect_counts.max()).unsqueeze(0).repeat(intersect_counts.shape[0],1).cuda() \
                                  < intersect_counts.unsqueeze(1)
        # get the dir follow the order of unique_intersect_voxel_idx
        # unique_intersect_voxel_idx is in ascending order
        sorted_indices_idx = torch.argsort(intersect_inverse_indices)
        sorted_segments_dir = masked_valid_segments_dir[sorted_indices_idx]
        sorted_scores = masked_valid_scores[sorted_indices_idx]

        # masked scatter to points
        scores_indicator = sorted_scores  # note: choose scores as indicator
        segments_dir_group = torch.zeros((unique_intersect_voxel_idx.shape[0], intersect_counts.max(), 3)).cuda()
        segments_dir_group = segments_dir_group.masked_scatter(
            segments_dir_group_mask.unsqueeze(-1).repeat(1, 1, sorted_segments_dir.shape[-1]),sorted_segments_dir)
        segments_scores_group = torch.zeros((unique_intersect_voxel_idx.shape[0], intersect_counts.max())).cuda()
        segments_scores_group = segments_scores_group.masked_scatter(segments_dir_group_mask, scores_indicator)
        self.unique_intersect_voxel_idx, self.segments_dir_group, self.segments_scores_group, self.segments_dir_group_mask = \
            unique_intersect_voxel_idx, segments_dir_group, segments_scores_group, segments_dir_group_mask

    def select_dir(self, line_similar_th):
        valid_mask = self.segments_dir_group_mask.any(dim=-1)
        # the max is follow the order of unique_intersect_voxel_idx
        intersect_highest_score_idx = torch.argmax(self.segments_scores_group, dim=1)
        voxel_num = self.segments_dir_group.shape[0]
        highest_score_segments_dir = self.segments_dir_group[torch.arange(voxel_num).cuda(), intersect_highest_score_idx]
        # not select the dir repeatedly
        highest_score_segments_dir[~valid_mask] = 0

        # get the weighted highest_score dir
        inner_segments_dir = (highest_score_segments_dir.unsqueeze(1) * self.segments_dir_group).sum(dim=-1)
        cos_dis_segments_dir = inner_segments_dir.abs()
        similar_mask = cos_dis_segments_dir > line_similar_th
        # change the similar dir to the same half plane of the selected highest_score dir
        change_dir_mask = inner_segments_dir < 0
        self.segments_dir_group[similar_mask * change_dir_mask] *= -1
        similar_dir_num = similar_mask.sum(dim=1, keepdim=True)
        segments_scores_group_tmp = self.segments_scores_group.clone()
        segments_scores_group_tmp[~similar_mask] = 0
        highest_score_segments_weight = segments_scores_group_tmp.sum(dim=1, keepdim=True)
        weighted_highest_score_segments_dir = (self.segments_dir_group * self.segments_scores_group.unsqueeze(-1)).sum(dim=1) / similar_dir_num
        weighted_highest_score_segments_dir = weighted_highest_score_segments_dir / weighted_highest_score_segments_dir.norm(dim=-1,keepdim=True)
        weighted_highest_score_segments_dir[weighted_highest_score_segments_dir.isnan()] = 0

        # remove the similar dir in self.segments_dir_group and self.segments_scores_group for the next dir selection
        self.segments_dir_group_mask[similar_mask] = False
        self.segments_scores_group[similar_mask] = 0
        self.segments_dir_group[similar_mask] = 0

        return weighted_highest_score_segments_dir, highest_score_segments_weight
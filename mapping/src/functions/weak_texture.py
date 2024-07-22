import torch

class weak_texture():
    def __init__(self, min_point_each_voxel, num_vertexes):
        self.min_point_each_voxel = min_point_each_voxel
        self.compress_simple_area_mask_voxel = torch.zeros(
            (num_vertexes, 1),
            requires_grad=False, dtype=torch.float32,
            device=torch.device("cuda")).bool()
        self.compress_color_edge_voxel = torch.zeros(
            (num_vertexes, 1),
            requires_grad=False, dtype=torch.float32,
            device=torch.device("cuda"))
        self.compress_simple_area_weight_voxel = torch.zeros(
            (num_vertexes, 1),
            requires_grad=False, dtype=torch.float32,
            device=torch.device("cuda"))


    def texture_pattern_determination(self, voxel_color_edge_mean, counts, structure_idx, inflate_num):
        if inflate_num != 0:
            noninflate_voxel_idx = structure_idx[:-inflate_num, 0].long()  # 除去膨胀的voxel
        else:
            noninflate_voxel_idx = structure_idx[:, 0].long()
        valid_count = counts[counts > self.min_point_each_voxel].unsqueeze(-1)
        valid_color_edge_mean = voxel_color_edge_mean[counts > self.min_point_each_voxel]
        origin_color_edge_mean = self.compress_color_edge_voxel[noninflate_voxel_idx]
        origin_compress_simple_area_weight_voxel = self.compress_simple_area_weight_voxel[noninflate_voxel_idx]
        weighted_color_edge_mean = (
                                           origin_color_edge_mean * origin_compress_simple_area_weight_voxel + valid_color_edge_mean * (
                                               valid_count / 10)) / \
                                   (
                                           origin_compress_simple_area_weight_voxel + valid_count / 10)  # avoid too large
        self.compress_color_edge_voxel[noninflate_voxel_idx] = weighted_color_edge_mean
        self.compress_simple_area_mask_voxel[noninflate_voxel_idx] = weighted_color_edge_mean < 0.2
        self.compress_simple_area_weight_voxel[noninflate_voxel_idx] += valid_count / 10  # avoid too large
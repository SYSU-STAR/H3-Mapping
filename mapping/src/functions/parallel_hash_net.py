import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

class Decoder(nn.Module):
    def __init__(self,
                 bound=None,
                 voxel_size=None,
                 L=None,
                 F_entry=None,
                 log2_T=None,
                 color_b=None,
                 sdf_b=None,
                 heterogeneous_grids=True,
                 **low_freq_directions_specs):
        super().__init__()

        self.bound = torch.FloatTensor(bound)
        self.bound_dis = self.bound[:, 1] - self.bound[:, 0]
        self.max_dis = torch.ceil(torch.max(self.bound_dis))
        N_min = int(self.max_dis / voxel_size)
        self.heterogeneous_grids = heterogeneous_grids
        self.hash_sdf_out = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=1,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "hash": "MultiPrime",
                    "n_levels": L,
                    "n_features_per_level": F_entry,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,  # 1/base_resolution is the grid_size
                    "per_level_scale": sdf_b,
                    "interpolation": "Linear"
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_la2yers": 1,
                }
            )

        self.hash_color_encoding = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "hash": "MultiPrime",
                    "n_levels": L,
                    "n_features_per_level": F_entry,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,  # 1/base_resolution is the grid_size
                    "per_level_scale": color_b,
                    "interpolation": "Linear"
                }
            )
        if self.heterogeneous_grids:
            mlp_input_dim = 2 + low_freq_directions_specs['max_low_freq_dir_num']
        else:
            mlp_input_dim = 1

        self.mlp_color = nn.Sequential(
            nn.Linear(mlp_input_dim * L * F_entry, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
        self.mlp_color.apply(init_weights)

    def get_color(self, xyz, map_states, chunk_samples=None):
        xyz = xyz.double()
        xyz = (xyz - self.bound[:, 0].to(xyz.device)) / self.bound_dis.to(xyz.device)
        if self.heterogeneous_grids:
            rgb = self.get_color_use_liner(xyz, map_states, chunk_samples)
        else:
            rgb_emb = self.hash_color_encoding(xyz)
            rgb = self.mlp_color(rgb_emb)
        return rgb

    def get_sdf(self, xyz):
        xyz = (xyz - self.bound[:, 0].to(xyz.device)) / self.bound_dis.to(xyz.device)
        sdf = self.hash_sdf_out(xyz)
        return sdf

    def get_color_use_liner(self, xyz, map_states, chunk_samples):
        compress_rot = map_states['compress_rot_info_voxel'][
            chunk_samples['sampled_point_voxel_idx'].long()]
        sampled_idx = chunk_samples["sampled_point_voxel_idx"].long()
        point_xyz = map_states["voxel_center_xyz"]
        point_xyz = F.embedding(sampled_idx, point_xyz)

        point_xyz_hash = (point_xyz - self.bound[:, 0].to(point_xyz.device)) / self.bound_dis.to(
            point_xyz.device)
        distance = xyz - point_xyz_hash

        compress_rate = 0.1
        compress_rot_tmp = compress_rot.clone()
        compress_rot_tmp[:, :, -1, :] *= compress_rate
        compress_feat_num = compress_rot_tmp.shape[1]
        valid_compress_rot_tmp_mask = compress_rot_tmp.isnan().any(dim=-1).any(dim=-1) == False
        compress_distance = distance.clone().unsqueeze(1).repeat(1, compress_feat_num, 1)
        compress_distance[valid_compress_rot_tmp_mask] = torch.einsum('tij,tj -> ti',
                                                                      compress_rot_tmp[
                                                                          valid_compress_rot_tmp_mask].double(),
                                                                      compress_distance[
                                                                          valid_compress_rot_tmp_mask])
        compress_point_xyz_hash = point_xyz_hash.clone().unsqueeze(1).repeat(1, compress_feat_num, 1)
        compress_point_xyz_hash[valid_compress_rot_tmp_mask] = torch.einsum('tij,tj -> ti',
                                                                            compress_rot[
                                                                                valid_compress_rot_tmp_mask],
                                                                            compress_point_xyz_hash[
                                                                                valid_compress_rot_tmp_mask]).float()
        rgb_emb1 = self.hash_color_encoding(
            point_xyz_hash + distance, prime_id=0)

        rgb_emb2 = self.hash_color_encoding(
            torch.tensor([compress_rate, compress_rate, compress_rate], device=xyz.device) * xyz, prime_id=1)

        compress_simple_area_mask = map_states["compress_simple_area_mask_voxel"][sampled_idx]
        mask_simple_area = torch.ones_like(rgb_emb2)
        mask_simple_area[~compress_simple_area_mask.squeeze()] = 0
        rgb_emb2 = rgb_emb2 * mask_simple_area

        feat_compress_all = []
        for i in range(compress_feat_num):
            feat_compress = self.hash_color_encoding(
                compress_point_xyz_hash[:, i] + compress_distance[:, i], prime_id=2 + i)
            mask_feat_compress = torch.ones_like(feat_compress)
            mask_feat_compress[~valid_compress_rot_tmp_mask[:, i]] = 0
            feat_compress_all.append(feat_compress * mask_feat_compress)

        feat_compress_all = torch.concatenate(feat_compress_all, dim=1)
        rgb_emb = torch.cat((rgb_emb1, rgb_emb2, feat_compress_all), dim=1)
        rgb = self.mlp_color(rgb_emb)
        return rgb

    def get_values(self, xyz, map_states, chunk_samples):
        xyz = xyz.double()
        xyz = (xyz - self.bound[:, 0].to(xyz.device)) / self.bound_dis.to(xyz.device)
        sdf = self.hash_sdf_out(xyz)
        if self.heterogeneous_grids:
            rgb = self.get_color_use_liner(xyz, map_states, chunk_samples)
        else:
            # raw
            rgb_emb = self.hash_color_encoding(xyz)
            rgb = self.mlp_color(rgb_emb)
        return sdf, rgb

    def forward(self, xyz, map_states, chunk_samples):
        sdf, rgb = self.get_values(xyz, map_states, chunk_samples)
        return {
            'color': rgb[:, :3],
            'sdf': sdf[:, :]
        }


if __name__ == "__main__":
    network = Decoder(1, 128, 16, skips=[], embedder='none', multires=0)
    print(network)

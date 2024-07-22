import torch
import torch.nn as nn
from S3IM.s3im import S3IM

class Criterion(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.rgb_weight = args.criteria["rgb_weight"]
        self.depth_weight = args.criteria["depth_weight"]
        self.sdf_weight = args.criteria["sdf_weight"]
        self.fs_weight = args.criteria["fs_weight"]
        self.truncation = args.criteria["sdf_truncation"]
        self.use_s3im_loss = args.criteria["use_s3im_loss"]
        self.max_dpeth = args.data_specs["max_depth"]
        self.sign_weight = args.criteria["sign_weight"]

    def forward(self, outputs, obs, use_color_loss=True,
                use_depth_loss=True, compute_sdf_loss=True,
                weight_depth_loss=False):

        img, depth = obs  # img:(1,N_rays,3) depth:(1,N_rays)
        loss = 0

        pred_depth = outputs["depth"]  # (N_rays)
        pred_color = outputs["color"]  # (N_rays,3)
        pred_sdf = outputs["sdf"]  # (N_rays,N_samples_every_ray)
        z_vals = outputs["z_vals"]  # (N_rays,N_samples_every_ray)
        ray_mask = outputs["ray_mask"]  # (1,N_rays)
        weights = outputs["weights"]  # (N_rays,N_samples_every_ray)
        self.mask_before_second_surface = outputs["mask_before_second_surface"]  # (N_rays,N_samples_every_ray)

        gt_depth = depth[ray_mask]  # (N_rays)
        gt_color = img[ray_mask]  # (N_rays,3)

        if use_depth_loss:
            valid_depth = (gt_depth > 0.01) & (gt_depth < self.max_dpeth)
            depth_loss = (gt_depth - pred_depth).abs()  # (N_rays)

            if weight_depth_loss:
                depth_var = weights * ((pred_depth.unsqueeze(-1) - z_vals) ** 2)
                depth_var = torch.sum(depth_var, -1)
                tmp = depth_loss / torch.sqrt(depth_var + 1e-10)
                valid_depth = (tmp < 10 * tmp.median()) & valid_depth
            depth_loss = depth_loss[valid_depth].mean()
            loss += self.depth_weight * depth_loss

        if use_color_loss:
            color_loss = (gt_color - pred_color).abs()[valid_depth].mean()
            loss += self.rgb_weight * color_loss

        if self.use_s3im_loss:
            self.s3im_func = S3IM(kernel_size=4, stride=4, repeat_time=10, patch_height=pred_color.shape[0] // 64,
                                  patch_width=64).cuda()
            loss += self.s3im_func(pred_color, gt_color)

        if compute_sdf_loss:
            fs_loss, sdf_loss, sign_fs_loss, sign_sdf_loss = self.get_sdf_loss(
                z_vals, gt_depth, pred_sdf,
                truncation=self.truncation,
                loss_type='l2'
            )
            loss += self.fs_weight * fs_loss
            loss += self.sdf_weight * sdf_loss

        return loss

    def compute_loss(self, x, y, mask=None, loss_type="l2"):
        if mask is None:
            mask = torch.ones_like(x).bool()
        if loss_type == "l1":
            return torch.mean(torch.abs(x - y)[mask])
        elif loss_type == "l2":
            return torch.mean(torch.square(x - y)[mask])

    def get_masks(self, z_vals, depth, epsilon):

        front_mask = torch.where(
            z_vals < (depth - epsilon),
            torch.ones_like(z_vals),
            torch.zeros_like(z_vals),
        )
        back_mask = torch.where(
            z_vals > (depth + epsilon),
            torch.ones_like(z_vals),
            torch.zeros_like(z_vals),
        )
        depth_mask = torch.where(
            (depth > 0.0) & (depth < self.max_dpeth), torch.ones_like(
                depth), torch.zeros_like(depth)
        )
        sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

        if self.mask_before_second_surface is not None:
            sdf_mask = sdf_mask * self.mask_before_second_surface

        num_fs_samples = torch.count_nonzero(front_mask).float()
        num_sdf_samples = torch.count_nonzero(sdf_mask).float()
        num_samples = num_sdf_samples + num_fs_samples
        fs_weight = 1.0 - num_fs_samples / num_samples
        sdf_weight = 1.0 - num_sdf_samples / num_samples

        return front_mask, back_mask, sdf_mask, fs_weight, sdf_weight

    def get_sdf_loss(self, z_vals, depth, predicted_sdf, truncation, loss_type="l2"):
        """
        compute sdf loss,contain about free-space loss(fs_loss) and SDF loss(sdf_loss)

        Args:
            z_vals (tensor,N_rays*N_samevrays_hitvox): the depth of sampling points
            depth (tensor,N_rays): gt depth corresponding to every rays
            predicted_sdf (tensor,N_rays*N_rays*N_samevrays_hitvox): predicted sdf of sampling points
            truncation (int): truncation value in tsdf.
            loss_type (str,Default="l2"): loss stytle
        
        Returns:
            fs_loss (tensor,1): free-space loss(fs_loss)
            sdf_loss (tensor,1): SDF loss(sdf_loss)
        """
        depth_tmp = depth.unsqueeze(-1).expand(*z_vals.shape).clone()
        front_mask, back_mask, sdf_mask, fs_weight, sdf_weight = self.get_masks(
            z_vals, depth_tmp, truncation
        )
        fs_loss = (self.compute_loss(predicted_sdf * front_mask, torch.ones_like(
            predicted_sdf) * front_mask, loss_type=loss_type, ) * fs_weight)
        sdf_loss = (self.compute_loss((z_vals + predicted_sdf * truncation) * sdf_mask,
                                      depth_tmp * sdf_mask,
                                      loss_type=loss_type, ) * sdf_weight)
        sign_fs_loss, sign_sdf_loss = 0, 0
        return fs_loss, sdf_loss, sign_fs_loss, sign_sdf_loss
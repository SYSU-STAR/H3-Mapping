import torch
import torch.nn as nn

@torch.no_grad()
def edge_conv2d(img_dim, device): # img: [B,3,H,W]
    by_kernal = torch.Tensor([
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]).to(device)
    by_kernal = by_kernal.repeat(img_dim,1,1)
    by_kernal = by_kernal.reshape((1, img_dim, 3, 3))

    conv_edge = nn.Conv2d(img_dim,img_dim,kernel_size=3,padding=1,bias=False,padding_mode='replicate')
    conv_edge.weight.data = by_kernal
    return conv_edge

@torch.no_grad()
def dist_conv2d(img_dim, device): # img: [B,3,H,W]
    by_kernal_x1 = torch.Tensor([
        [[0, 0, 0], [-1, 1, 0], [0, 0, 0]]]).to(device)
    by_kernal_x1 = by_kernal_x1.repeat(img_dim,1,1)
    by_kernal_x1 = by_kernal_x1.reshape((1, img_dim, 3, 3))

    by_kernal_x2 = torch.Tensor([
        [[0, 0, 0], [0, 1, -1], [0, 0, 0]]]).to(device)
    by_kernal_x2 = by_kernal_x2.repeat(img_dim,1,1)
    by_kernal_x2 = by_kernal_x2.reshape((1, img_dim, 3, 3))

    by_kernal_y1 = torch.Tensor([
        [[0, -1, 0], [0, 1, 0], [0, 0, 0]]]).to(device)
    by_kernal_y1 = by_kernal_y1.repeat(img_dim,1,1)
    by_kernal_y1 = by_kernal_y1.reshape((1, img_dim, 3, 3))

    by_kernal_y2 = torch.Tensor([
        [[0, 0, 0], [0, 1, 0], [0, -1, 0]]]).to(device)
    by_kernal_y2 = by_kernal_y2.repeat(img_dim,1,1)
    by_kernal_y2 = by_kernal_y2.reshape((1, img_dim, 3, 3))

    by_kernal_diag1 = torch.Tensor([
        [[-1, 0, 0], [0, 1, 0], [0, 0, 0]]]).to(device)
    by_kernal_diag1 = by_kernal_diag1.repeat(img_dim,1,1)
    by_kernal_diag1 = by_kernal_diag1.reshape((1, img_dim, 3, 3))

    by_kernal_diag2 = torch.Tensor([
        [[0, 0, 0], [0, 1, 0], [0, 0, -1]]]).to(device)
    by_kernal_diag2 = by_kernal_diag2.repeat(img_dim,1,1)
    by_kernal_diag2 = by_kernal_diag2.reshape((1, img_dim, 3, 3))

    by_kernal_inv_diag1 = torch.Tensor([
        [[0, 0, -1], [0, 1, 0], [0, 0, 0]]]).to(device)
    by_kernal_inv_diag1 = by_kernal_inv_diag1.repeat(img_dim,1,1)
    by_kernal_inv_diag1 = by_kernal_inv_diag1.reshape((1, img_dim, 3, 3))

    by_kernal_inv_diag2 = torch.Tensor([
        [[0, 0, 0], [0, 1, 0], [-1, 0, 0]]]).to(device)
    by_kernal_inv_diag2 = by_kernal_inv_diag2.repeat(img_dim,1,1)
    by_kernal_inv_diag2 = by_kernal_inv_diag2.reshape((1, img_dim, 3, 3))

    by_kernal = torch.cat((by_kernal_x1,by_kernal_x2, by_kernal_y1, by_kernal_y2, by_kernal_diag1, by_kernal_diag2, by_kernal_inv_diag1, by_kernal_inv_diag2), dim=0)

    conv_dist = nn.Conv2d(img_dim, img_dim, kernel_size=3, padding=1,bias=False,padding_mode='replicate')
    conv_dist.weight.data = by_kernal
    conv_dist.requires_grad_(False)

    return conv_dist
import torch
import torch_scatter


def sampling_without_replacement(logp, k):
    def gumbel_like(u):
        return -torch.log(-torch.log(torch.rand_like(u) + 1e-7) + 1e-7)

    scores = logp + gumbel_like(logp)
    return scores.topk(k, dim=-1)[1]

"""
sample rays from image
"""

def sample_rays(mask, num_samples):
    B, H, W = mask.shape
    mask_unfold = mask.reshape(-1)
    indices = torch.rand_like(mask_unfold).topk(num_samples)[1]
    sampled_masks = (torch.zeros_like(
        mask_unfold).scatter_(-1, indices, 1).reshape(B, H, W) > 0)
    return sampled_masks

def updownsampling_voxel(points, indices, counts, mode):
    bin_size = counts.shape[0]
    if len(points.shape) == 1:
        # elements should be scattered, element channel
        remain_elements = torch.zeros(bin_size).cuda()
        if mode == 'max':
            summed_elements, argmax = torch_scatter.scatter_max(dim=0, index=indices, src=points)
            remain_elements[:summed_elements.shape[0]] = summed_elements
            return remain_elements, argmax
        elif mode == 'min':
            summed_elements, argmin = torch_scatter.scatter_min(dim=0, index=indices, src=points)
            remain_elements[:summed_elements.shape[0]] = summed_elements
            return remain_elements, argmin
        elif mode in ['mean', 'sum']:
            summed_elements = torch_scatter.scatter(dim=0, index=indices, src=points, reduce=mode)
            remain_elements[:summed_elements.shape[0]] = summed_elements
            return remain_elements
        else:
            raise NotImplementedError
    elif len(points.shape) == 2:
        elements, channels = points.shape
        # elements should be scattered, element channel
        remain_elements = torch.zeros(bin_size, channels).cuda()
        if mode == 'max':
            summed_elements, argmax = torch_scatter.scatter_max(dim=0, index=indices, src=points)
            remain_elements[:summed_elements.shape[0], :] = summed_elements
            return remain_elements, argmax
        elif mode == 'min':
            summed_elements, argmin = torch_scatter.scatter_min(dim=0, index=indices, src=points)
            remain_elements[:summed_elements.shape[0], :] = summed_elements
            return remain_elements, argmin
        elif mode in ['mean', 'sum']:
            summed_elements = torch_scatter.scatter(dim=0, index=indices, src=points, reduce=mode)
            remain_elements[:summed_elements.shape[0], :] = summed_elements
            return remain_elements
        else:
            raise NotImplementedError
    elif len(points.shape) == 3:
        # batch, elements should be scattered, element channel
        batch, elements, channels = points.shape
        remain_elements = torch.zeros(batch, bin_size, channels).cuda()
        if mode == 'max':
            summed_elements, argmax = torch_scatter.scatter_max(dim=1, index=indices, src=points)
            remain_elements[:, :summed_elements.shape[1], :] = summed_elements
            return remain_elements, argmax
        elif mode == 'min':
            summed_elements, argmin = torch_scatter.scatter_min(dim=1, index=indices, src=points)
            remain_elements[:, :summed_elements.shape[1], :] = summed_elements
            return remain_elements, argmin
        elif mode in ['mean', 'sum']:
            summed_elements = torch_scatter.scatter(dim=1, index=indices, src=points, reduce=mode)
            remain_elements[:, :summed_elements.shape[1], :] = summed_elements
            return remain_elements
        else:
            raise NotImplementedError
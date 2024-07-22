import torch
from torch.nn.utils.rnn import pad_sequence

"""
    Overwrite all voxels contained in the keyframe multiple times
    
    Note that there is a related implementation outside of this function. 
    In the insert_keyframe class method in mapping, The function is to add the new voxels contained 
    in each newly added key frame to the set of unoptimized voxels. 
    Cover all the voxels contained in the key frame multiple times
    Args:
        kf_graph (list, l): Save information about keyframes into a list.
        kf_seen_voxel_num (list, l): This is a list, each element is the number of the corresponding 
                                    voxels contained in the keyframe in kf_graph
        unoptimized_voxels (tensor, N*3):  Coordinates of all unoptimized voxels  
        optimized_voxels (tensor, M*3): Coordinates of all optimized voxels
        windows_size (int, w): Number of keyframe pools
    Returns:
        target_graph (list, w): The keyframe of the final selected w
        unoptimized_voxels (tensor, N'*3): The voxel seen through the selected keyframe, 
                                        the updated unoptimized voxel coordinates
        optimized_voxels (tensor, M'*3): Voxels seen through selected keyframes, 
                                    updated optimized voxel coordinates
                
"""


def multiple_max_set_coverage(kf_graph, kf_seen_voxel_num, kf_unoptimized_voxels, kf_optimized_voxels,
                              windows_size, kf_svo_idx, kf_all_voxels, num_vertexes, kf_voxel_freq,
                              round_num, latest_select_round, insert_round, kf_seen_voxel, kf_voxel_weight):
    cnt = 0
    target_graph = []
    padded_tensor = pad_sequence(kf_svo_idx, batch_first=True, padding_value=-1)[:, :, 0]
    if len(kf_voxel_weight) != 0:
        padded_weight = pad_sequence(kf_voxel_weight, batch_first=True, padding_value=-1)[:, :, 0]
    if kf_unoptimized_voxels is None:
        kf_unoptimized_voxels = torch.zeros(num_vertexes).cuda().bool()  # unoptimized voxels
        kf_optimized_voxels = torch.zeros(
            num_vertexes).cuda().bool()  # The voxels seen by the currently selected x keyframes

        kf_unoptimized_voxels[padded_tensor.long() + 1] += True  # Empty number 0, because the back pad is filled with 0
        kf_unoptimized_voxels[0] = False
        if len(kf_voxel_weight) == 0:
            kf_seen_voxel_num = torch.tensor(kf_seen_voxel_num)  # (N)
            value, index = torch.max(kf_seen_voxel_num, dim=0)
        else:
            # weighted by edge observation
            result_num = (kf_unoptimized_voxels[padded_tensor.long() + 1].float() * padded_weight).sum(-1)
            value, index = torch.max(result_num, dim=0)

        target_graph += [kf_graph[index]]
        kf_unoptimized_voxels[kf_svo_idx[index].long() + 1] *= False  # Empty number 0
        kf_optimized_voxels[kf_svo_idx[index].long() + 1] += True
        cnt += 1
        if len(latest_select_round) != 0:
            latest_select_round[index] = round_num
    while cnt != int(windows_size):
        if len(kf_voxel_weight) == 0:
            result_num = kf_unoptimized_voxels[padded_tensor.long() + 1].sum(-1)  # Empty number 0
            value, index = torch.max(result_num, dim=0)
        else:
            # weighted by edge observation
            result_num = (kf_unoptimized_voxels[padded_tensor.long() + 1].float() * padded_weight).sum(-1)
            value, index = torch.max(result_num, dim=0)

        if len(latest_select_round) != 0:
            latest_select_round[index] = round_num

        target_graph += [kf_graph[index]]
        kf_unoptimized_voxels[kf_svo_idx[index].long() + 1] *= False
        kf_optimized_voxels[kf_svo_idx[index].long() + 1] += True
        cnt += 1
        if kf_unoptimized_voxels.any() == False:  # If all are optimized
            if len(latest_select_round) != 0:
                latest_select_round_tensor = torch.tensor(latest_select_round).cuda()
                insert_round_tensor = torch.tensor(insert_round).cuda()
                remove_mask = (insert_round_tensor + 1 <= round_num) & (latest_select_round_tensor + 1 <= round_num)
                remove_idx_all = remove_mask.nonzero()
                # avoid the index change when del
                remove_idx_all = torch.sort(remove_idx_all, descending=True, dim=0)[0]
                if len(remove_idx_all) != 0:
                    for remove_idx in remove_idx_all:
                        del kf_graph[remove_idx]
                        del kf_seen_voxel[remove_idx]
                        del kf_seen_voxel_num[remove_idx]
                        del kf_svo_idx[remove_idx]
                        del insert_round[remove_idx]
                        del latest_select_round[remove_idx]
                        if len(kf_voxel_weight) != 0:
                            del kf_voxel_weight[remove_idx]
                    padded_tensor = pad_sequence(kf_svo_idx, batch_first=True, padding_value=-1)[:, :, 0]
                    if len(kf_voxel_weight) != 0:
                        padded_weight = pad_sequence(kf_voxel_weight, batch_first=True, padding_value=-1)[:, :, 0]
            kf_all_voxels[padded_tensor.long() + 1] += True
            kf_all_voxels[0] = False  # Empty number 0
            # Unoptimized voxels = all voxels that need to be optimized - \
            # voxels seen by the currently selected x keyframes
            if len(latest_select_round) == 0:
                kf_unoptimized_voxels = kf_all_voxels * ~kf_optimized_voxels
            else:
                kf_unoptimized_voxels = kf_all_voxels # if the removed one has some unique,
            round_num += 1
        kf_optimized_voxels *= False # Only count the voxels selected in the current iteration; this is used to exclude the voxels selected in the current round when all voxels have been selected once.
    return target_graph, kf_unoptimized_voxels, kf_optimized_voxels, kf_all_voxels, kf_voxel_freq, round_num, latest_select_round, insert_round, kf_seen_voxel, kf_voxel_weight


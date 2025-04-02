from loguru import logger
import torch
from einops import repeat
from kornia.utils import create_meshgrid

##############  ↓  Coarse-Level supervision  ↓  ##############

@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt

@torch.no_grad()
def spvs_coarse(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
        }
    """
    if not isinstance(data['image0'], torch.Tensor) or not isinstance(data['image1'], torch.Tensor):
        raise TypeError("Expected 'image0' and 'image1' to be tensors")

    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['LOFTR']['RESOLUTION'][0]
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0 * w0, 2).repeat(N, 1, 1)  # [N, hw, 2]
    grid_pt0_i = grid_pt0_c * scale
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1 * w1, 2).repeat(N, 1, 1)
    grid_pt1_i = grid_pt1_c * scale

    conf_matrix_gt = torch.zeros(N, h0 * w0, h1 * w1, device=device)

    for b in range(N):
        matching_matrix = data['matching_matrix'][b]

        for y0 in range(matching_matrix.shape[0]):
            for x0 in range(matching_matrix.shape[1]):
                x1, y1 = matching_matrix[y0, x0]
                if x1 > 0 and y1 > 0:
                    i_idx = torch.tensor((y0 // scale) * w0 + (x0 // scale), device=device)
                    j_idx = torch.tensor((y1 // scale) * w1 + (x1 // scale), device=device)
                    if i_idx < h0 * w0 and j_idx < h1 * w1:
                        conf_matrix_gt[b, i_idx.long(), j_idx.long()] = 1
                    # This gets triggered very very often, so come back and fix it later.
                    # else:
                    #     logger.warning(f"Index out of bounds: i_idx={i_idx}, j_idx={j_idx}, h0*w0={h0 * w0}, h1*w1={h1 * w1}")

    data.update({'conf_matrix_gt': conf_matrix_gt})

    b_ids, i_ids, j_ids = torch.where(conf_matrix_gt > 0)
    data.update({'spv_b_ids': b_ids, 'spv_i_ids': i_ids, 'spv_j_ids': j_ids})

def compute_supervision_coarse(data, config):
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth', 'custom', 'train', 'val', 'test']:
        spvs_coarse(data, config)
    else:
        raise ValueError(f'Unknown data source: {data_source}')

##############  ↓  Fine-Level supervision  ↓  ##############

@torch.no_grad()
def spvs_fine(data, config):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # Ensure data['image0'] is a tensor
    if not isinstance(data['image0'], torch.Tensor):
        raise TypeError("Expected 'image0' to be a tensor")

    scale = config['LOFTR']['RESOLUTION'][1]
    radius = config['LOFTR']['FINE_WINDOW_SIZE'] // 2

    b_ids, i_ids, j_ids = data['spv_b_ids'], data['spv_i_ids'], data['spv_j_ids']

    expec_f_gt = torch.zeros(len(b_ids), 2, device=data['image0'].device)

    for idx, (b, i, j) in enumerate(zip(b_ids, i_ids, j_ids)):
        matching_matrix = data['matching_matrix'][b]
        if i >= matching_matrix.shape[0] or j >= matching_matrix.shape[1]:  # Ensure index is within bounds
            continue
        x1, y1 = matching_matrix[i // matching_matrix.shape[1], i % matching_matrix.shape[1]]
        expec_f_gt[idx] = (torch.tensor([i % matching_matrix.shape[1], i // matching_matrix.shape[1]], device=data['image0'].device) - torch.tensor([x1, y1], device=data['image0'].device)) / scale / radius  # [M, 2]

    data.update({"expec_f_gt": expec_f_gt})

def compute_supervision_fine(data, config):
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth', 'custom', 'train', 'val', 'test']:
        spvs_fine(data, config)
    else:
        raise NotImplementedError

import torch
import numpy as np
from collections import deque
from scipy.ndimage import gaussian_filter

def get_bfs_centers(image, num_elements, height, width):
    scores = torch.flatten(image)
    flattened_scores, flattened_indices = torch.topk(scores, num_elements)

    top_row_indices = []
    for it in range(0, num_elements, 1):
        local_row = flattened_indices[it].item() // width
        local_col = flattened_indices[it].item() % width
        top_row_indices.append([local_row, local_col])

    return top_row_indices
    
def bfs(image, visited, i, j, cluster, threshold, max_search_radius):
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    queue = deque([(i, j)])
    search_radius = deque([0])
    visited[i][j] = True
    
    while queue:
        ci, cj = queue.popleft()
        curr_radius = search_radius.popleft()
        cluster.append([ci, cj])
        
        for direction in directions:
            ni, nj = ci + direction[0], cj + direction[1]
            if 0 <= ni < image.shape[0] and 0 <= nj < image.shape[1] and not visited[ni][nj] and image[ni][nj] >= threshold and curr_radius < max_search_radius:
                visited[ni][nj] = True
                queue.append((ni, nj))
                search_radius.append(curr_radius + 1)

def circum_bounding_box(cluster):
    '''
    Function to make the bounding box over clusters
    '''
    min_height, max_height, min_width, max_width = (1e4, 0, 1e4, 0)
    for patch in cluster:
        if patch[0] < min_height : min_height = patch[0]
        if patch[0] > max_height : max_height = patch[0]
        if patch[1] < min_width : min_width = patch[1]
        if patch[1] > max_width : max_width = patch[1]

    return ((min_height, min_width), (max_height, max_width))

def get_mines(image_attn_values : torch.Tensor,
                    num_height_patches : int,
                    num_width_patches : int,
                    min_patch_for_mine : int = 4, 
                    max_image_mines : int = 3, 
                    max_search_radius : int = 8,
                    cluster_thresh_frac :float = 0.7,
                    apply_gaussian_blur : int = 1):
    '''
    This function is used to get all the image mines
    (Args)
    image_attn_values (torch.Tensor) : The image attention values
    num_height_patches (int) : The number of patches in the height dimension after sub-crops
    num_width_patches (int) : The number of patches in the width dimension after sub-crops
    min_patch_for_mine (int, optional) : Minimum number of patches that can constitute a mine
    max_image_mines (int, optional) : Maximum number of image mines that can be formed
    max_search_radius (int, optional) : The maximum radius while performing breadth first search
    cluster_thresh_frac (float, optional) : The threshold for being considered for bfs as a part of that cluster
    '''

    if type(image_attn_values) is not torch.Tensor:
        image_attn_values = torch.tensor(image_attn_values)

    if apply_gaussian_blur == 1:
        image_np_array = image_attn_values.numpy()
        blurred_np_array = gaussian_filter(image_np_array, sigma=1)  # Adjust sigma as needed
        image_attn_values = torch.from_numpy(blurred_np_array)
    elif apply_gaussian_blur == 2:
        image_np_array = image_attn_values.numpy()
        min_val = image_np_array.min()
        max_val = image_np_array.max()
        normalized_np_array = (image_np_array - min_val) / (max_val - min_val)
        image_attn_values = torch.from_numpy(normalized_np_array)

    num_centers = 10
    cluster_centers = get_bfs_centers(image_attn_values, num_centers, num_height_patches, num_width_patches)

    visited = torch.zeros_like(image_attn_values, dtype=bool)
    clusters = []

    for cluster_center in cluster_centers:
        i = cluster_center[0]
        j = cluster_center[1]
        cluster_thresh = image_attn_values[i][j] * cluster_thresh_frac

        if not visited[i][j]:
            cluster = []
            bfs(image_attn_values, visited, i, j, cluster, cluster_thresh, max_search_radius)
            if len(cluster) >= min_patch_for_mine:
                clusters.append(cluster)
    
    if len(clusters) > max_image_mines:
        clusters = clusters[ : max_image_mines]

    bounding_boxes = []
    for cluster in clusters:
        cluster_box = circum_bounding_box(cluster)
        bounding_boxes.append(cluster_box)
    
    return bounding_boxes, clusters

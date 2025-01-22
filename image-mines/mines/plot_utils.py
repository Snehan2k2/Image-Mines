import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import os

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import argparse
import pandas as pd
import pickle

# Load pre-trained CLIP model and processor
class SigSimilarity():
    def __init__(self, model_id = "google/siglip-so400m-patch14-384"):
        self.model = AutoModel.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def compute_similarity(self, sub_crops, answer):
        # Preprocess images and the question    
        images = sub_crops
        inputs = self.processor(text=[answer], images=images, return_tensors="pt", padding=True)

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Compute cosine similarity
        image_embeddings = outputs.image_embeds
        text_embeddings = outputs.text_embeds
        
        # Normalize embeddings
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        # Calculate cosine similarities
        similarities = (image_embeddings @ text_embeddings.T).squeeze().cpu().numpy()
        
        return similarities

def merge_boxes(boxes):
    # Sort the boxes based on the top-left corner (min_height, min_width)
    boxes = sorted(boxes, key=lambda box: (box[0][0], box[0][1]))
    merged_boxes = []

    for box in boxes:
        merged = False

        # Check against all merged boxes for potential overlaps
        new_merged_boxes = []

        for last_box in merged_boxes:
            if (box[0][0] <= last_box[1][0] and  # min_height of current box <= max_height of last box
                box[0][1] <= last_box[1][1] and  # min_width of current box <= max_width of last box
                box[1][0] >= last_box[0][0] and  # max_height of current box >= min_height of last box
                box[1][1] >= last_box[0][1]):    # max_width of current box >= min_width of last box
                # Merge the boxes
                box = (
                    (min(last_box[0][0], box[0][0]), min(last_box[0][1], box[0][1])),  # new min corner
                    (max(last_box[1][0], box[1][0]), max(last_box[1][1], box[1][1]))   # new max corner
                )
                merged = True  # Mark that we've merged
            else:
                new_merged_boxes.append(last_box)  # No overlap, keep the last box

        if not merged:
            new_merged_boxes.append(box)  # Add the current box if no merge occurred
        else:
            new_merged_boxes.append(box)  # Include the merged box

        merged_boxes = new_merged_boxes  # Update merged_boxes list

        # Continue merging until no more merges can occur
        while True:
            merged_any = False
            new_boxes = []
            skip_next = False
            
            for i in range(len(merged_boxes)):
                if skip_next:
                    skip_next = False
                    continue
                
                box_a = merged_boxes[i]
                merged = False
                
                for j in range(i + 1, len(merged_boxes)):
                    box_b = merged_boxes[j]
                    if (box_a[0][0] <= box_b[1][0] and  # min_height of box_a <= max_height of box_b
                        box_a[0][1] <= box_b[1][1] and  # min_width of box_a <= max_width of box_b
                        box_a[1][0] >= box_b[0][0] and  # max_height of box_a >= min_height of box_b
                        box_a[1][1] >= box_b[0][1]):    # max_width of box_a >= min_width of box_b
                        # Merge the two boxes
                        new_box = (
                            (min(box_a[0][0], box_b[0][0]), min(box_a[0][1], box_b[0][1])),  # new min corner
                            (max(box_a[1][0], box_b[1][0]), max(box_a[1][1], box_b[1][1]))   # new max corner
                        )
                        new_boxes.append(new_box)  # Add the merged box
                        merged = True  # Mark that we've merged
                        merged_any = True  # A merge occurred, so we'll need to check again
                        skip_next = True  # Skip the next box as it has been merged
                        break  # Exit the inner loop

                if not merged:
                    new_boxes.append(box_a)  # No merge, keep the box
            
            merged_boxes = new_boxes  # Update the merged boxes list
            if not merged_any:  # No more merges, break the loop
                break

    return merged_boxes

def make_3d_grid(layer_attn, height_view, width_view):
    matrix = layer_attn.view(height_view, width_view)
    grid = matrix.cpu().transpose(0,1).numpy()
    x, y = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]))

    # Create a 3D plot
    fig = plt.figure(figsize = (16, 16))
    ax = fig.add_subplot(111, projection='3d')
    norm = plt.Normalize(grid.min(), grid.max())
    colors = plt.cm.viridis(norm(grid))

    # Plot the bars
    ax.bar3d(x.flatten(), y.flatten(), np.zeros_like(grid.flatten()), 1, 1, grid.flatten(), color=colors.reshape(-1, 4))

    # Add labels and title
    ax.set_xlabel('Height')
    ax.set_ylabel('Width')
    ax.set_zlabel('Values')
    ax.set_title('Attention Weights vs Pixel Coordinate')

    plt.show()

def make_image_grids(image_path, clusters, img_config, output_path):
    img = Image.open(image_path)
    # img_config would be be matrix of r x c
    num_height_patches, num_width_patches = img_config
    scaler = img.width // num_width_patches # In the width dimension

    img = img.resize((num_width_patches * scaler, num_height_patches * scaler))
    draw = ImageDraw.Draw(img)

    # Determining the block sizes
    x_block_size = scaler #column wise
    y_block_size = img.height // num_height_patches #row wise

    # Draw black outlines over the specified cluster indices
    for cluster in clusters:
        outline_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for index in cluster:
            y, x = index
            top_left = (x * x_block_size, y * y_block_size)
            bottom_right = ((x + 1) * x_block_size, (y + 1) * y_block_size)
            draw.rectangle([top_left, bottom_right], outline=outline_color, width=1)
    
    img.save(output_path)

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def scaled_boxes(img, bounding_boxes, img_config, delta = 5):

    # img_config would be be matrix of r x c
    num_height_patches, num_width_patches = img_config
    scaler = img.width // num_width_patches # In the width dimension
    img = img.resize((num_width_patches * scaler, num_height_patches * scaler))

    # Determining the block sizes
    x_block_size = scaler # column wise
    y_block_size = img.height // num_height_patches # row wise

    boxes = []
    
    for idx, bounding_box in enumerate(bounding_boxes):
        
        ((min_height, min_width), (max_height, max_width)) = bounding_box
        top_left_x = min_width * x_block_size
        top_left_y = min_height * y_block_size
        bottom_right_x = (max_width + 1) * x_block_size
        bottom_right_y = (max_height + 1) * y_block_size
        
        coordinates = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        coordinates[0] = max(0, coordinates[0] - delta)
        coordinates[1] = max(0, coordinates[1] - delta)
        coordinates[2] = min(img.size[0], coordinates[2] + delta)
        coordinates[3] = min(img.size[1], coordinates[3] + delta)

        boxes.append(((coordinates[1], coordinates[0]), (coordinates[3], coordinates[2])))


    return merge_boxes(boxes)


def draw_blocks(img, bounding_boxes, img_config, output_folder_path, iter, answer, upscale_crops = False, delta = 10, siglip = False, draw_bool = True):

    # img_config would be be matrix of r x c
    bounding_boxes = scaled_boxes(img, bounding_boxes, img_config, delta = delta)
    num_height_patches, num_width_patches = img_config
    scaler = img.width // num_width_patches # In the width dimension
    img = img.resize((num_width_patches * scaler, num_height_patches * scaler))
    img_dup = img.copy()
    
    extension = '.jpg'
    ensure_directory_exists(os.path.join(output_folder_path, str(iter)))
    if draw_bool:
        draw = ImageDraw.Draw(img)

    if img.mode in ['RGBA', 'P', 'LA', 'CMYK', 'L']:
        img = img.convert('RGB')

    sig_model = SigSimilarity()

    similarity_score_base = -1

    if siglip:
        similarity_score_base = sig_model.compute_similarity(img, answer)
    
    for idx, bounding_box in enumerate(bounding_boxes):
        flag = -1
        ((min_height, min_width), (max_height, max_width)) = bounding_box
        top_left_x = min_width
        top_left_y = min_height
        bottom_right_x = max_width
        bottom_right_y = max_height
        
        coordinates = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

        if siglip:
            cropped_image = img_dup.crop(coordinates)

            if cropped_image.mode in ['RGBA', 'P', 'LA', 'CMYK', 'L']:
                cropped_image = cropped_image.convert('RGB')

            similarity_scores = sig_model.compute_similarity(cropped_image, answer)
            
            if similarity_scores >= 0.7*similarity_score_base:
                flag = flag + 2
                if draw_bool:
                    draw.rectangle(coordinates, outline="red", width=2)

        else:
            cropped_image = img_dup.crop(coordinates)

            if cropped_image.mode in ['RGBA', 'P', 'LA', 'CMYK', 'L']:
                cropped_image = cropped_image.convert('RGB')
            if draw_bool:
                draw.rectangle(coordinates, outline="red", width=2)

        # Upscaling the crops if required
        if upscale_crops:
            if cropped_image.width < 336:
                crop_scaler = 336 // cropped_image.width
            else:
                crop_scaler = 1
            new_crop_width = cropped_image.width * crop_scaler
            new_crop_height = cropped_image.height * crop_scaler
            cropped_image = cropped_image.resize((new_crop_width, new_crop_height))

        if siglip:
            if flag == 1:
                crop_save_location = os.path.join(output_folder_path, str(iter), f"sub_crop_{idx}{extension}")
                cropped_image.save(crop_save_location)
        else:
            crop_save_location = os.path.join(output_folder_path, str(iter), f"sub_crop_{idx}{extension}")
            cropped_image.save(crop_save_location)

    img_save_location = os.path.join(output_folder_path, str(iter), f"base{extension}")
    img.save(img_save_location)

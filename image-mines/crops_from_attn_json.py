import argparse
import io
import pandas as pd
import pickle
from PIL import Image, ImageDraw
from tqdm import tqdm
import json

from complexity.image_mines import get_mines
from complexity.plot_utils import draw_blocks

def load_image_from_path(img_path):
    img = Image.open(img_path)
    return img

def load_image(img_data):
    img = Image.open(io.BytesIO(img_data['bytes']))
    return img

def load_parquet(file_path : str, cols = ['question', 'image', 'answers']):
    data = pd.read_parquet(file_path)
    data = data[cols]
    return data

def load_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
def load_json(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)

def eval(args):
    data = load_json(args.json_path)
    attn_values = load_pkl(args.attn_file)

    for i in tqdm(range(500)):
        img = load_image_from_path(args.folder_path + '/' + data[i]['imgname'])
        image_config = attn_values[i][0].shape
        image_attn = attn_values[i][0]
        bounding_boxes, clusters = get_mines(image_attn, image_config[0], image_config[1], min_patch_for_mine=4, max_image_mines = 3, max_search_radius=min(image_config[0], image_config[1])/2, cluster_thresh_frac=0.2, apply_gaussian_blur=1)
        draw_blocks(img, bounding_boxes, image_config, args.output_path, i, attn_values[i][1], upscale_crops=True, delta=15, siglip=True, draw_bool=True)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str, default='/home/FRACTAL/snehan.j/image-mines/crops/COCO/coco_final.json')
    parser.add_argument("--output-path", type=str, default='/home/FRACTAL/snehan.j/crops/coco_crops')
    parser.add_argument("--attn-file", type=str, default='/home/FRACTAL/snehan.j/LLaVA-NeXT/attn-eval-final/coco.pkl')
    parser.add_argument("--folder-path", type=str, default='/home/FRACTAL/snehan.j/image-mines/crops/COCO/images')
    args = parser.parse_args()
    eval(args)

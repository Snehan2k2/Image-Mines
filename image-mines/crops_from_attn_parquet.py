import argparse
import io
import pandas as pd
import pickle
from PIL import Image, ImageDraw
from tqdm import tqdm

from complexity.image_mines import get_mines
from complexity.plot_utils import draw_blocks

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
    
def eval(args):
    df = load_parquet(args.parquet_path)
    attn_values = load_pkl(args.attn_file)

    idx = 0
    for _, row in tqdm(df.iloc[0:500].iterrows(), total=500):
        img = load_image(row['image'])
        image_config = attn_values[idx][0].shape
        image_attn = attn_values[idx][0]
        bounding_boxes, clusters = get_mines(image_attn, image_config[0], image_config[1], min_patch_for_mine=4, max_image_mines = 3, max_search_radius=min(image_config[0], image_config[1])/2, cluster_thresh_frac=0.2, apply_gaussian_blur=1)
        draw_blocks(img, bounding_boxes, image_config, args.output_path, idx, attn_values[idx][1], upscale_crops=True, delta=15, siglip=False, draw_bool=True)
        idx = idx + 1
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-path", type=str, default='/home/FRACTAL/kalash.shah/STRIM_codebase/dataset/VizWiz/3.parquet')
    parser.add_argument("--output-path", type=str, default='/home/FRACTAL/snehan.j/hyperparameters/vizwiz_crops_w_box')
    parser.add_argument("--attn-file", type=str, default='/home/FRACTAL/snehan.j/LLaVA-NeXT/attn-eval/vizwiz.pkl')
    args = parser.parse_args()
    eval(args)

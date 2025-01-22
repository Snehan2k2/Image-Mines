from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
import os 
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from PIL import Image
import psutil
import requests
import copy
import torch
import argparse
import json
from tqdm import tqdm
import time
from multiprocessing import Process
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer
import pickle

def make_heatmap(layer_attn, height_view, width_view):
    
    matrix = layer_attn.view(height_view, width_view)
    matrix = matrix.cpu().numpy()
    return matrix

def load_json(data, folder_path):
    queries = []
    img_paths = []
    for i in data:
        queries.append(i['query'])
        img_paths.append(folder_path + "/" + i['imgname'])

    return queries, img_paths

def eval(qs, img_path, args, final, idx):

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, attn_implementation="eager")

    url = img_path
    image = Image.open(url)

    device = "cuda"
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "llava_llama_3"
    user_prompt = qs
    print(user_prompt)
    question = DEFAULT_IMAGE_TOKEN + "\n {}".format(user_prompt) + "\n Answer briefly in only one sentence"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]

    with torch.inference_mode():
        config, output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=True,
            include_base_only=False,
            filter_newline=True)
        

    outputs = tokenizer.batch_decode(output_ids['sequences'], skip_special_tokens=True)[0]
    print(outputs)

    #Ignoring /n and <eot> tokens for the computation
    sub_crop_height = config[-2]
    sub_crop_width = config[-1]
    cum_token_averaged_attn = torch.zeros([1, sub_crop_height * sub_crop_width], device = "cpu")
    img_token_idx = (input_ids == -200).nonzero(as_tuple=True)[1].item()

    for output_attn in output_ids['attentions'][1:]:
        stacked_tensors = torch.stack(output_attn).cpu()
        stacked_tensors = stacked_tensors[24:32]
        mean_values = stacked_tensors.mean(dim=0)
        mean_values = mean_values.squeeze()
        average_values = mean_values.mean(dim=0)
        cum_token_averaged_attn += average_values[img_token_idx + 576: img_token_idx + 576 + sub_crop_height * sub_crop_width]

    cum_token_averaged_attn = cum_token_averaged_attn/(len(output_ids['attentions'][1:]))

    matrix = make_heatmap(cum_token_averaged_attn, sub_crop_height, sub_crop_width)
    final.append((matrix, outputs, idx))
    

def eval_main(args):

    with open(args.json_path, 'r') as file:
        data = json.load(file)

    data = data[:500]
    queries, img_paths = load_json(data, args.folder_path)
    final = []

    for i in tqdm(range(len(queries))):
        p = Process(target=eval(queries[i], img_paths[i], args, final, i))
        p.start()
        p.join()
    
    with open(args.pkl_path, 'wb') as f:
        pickle.dump(final, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="lmms-lab/llama3-llava-next-8b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--json-path", type=str)
    parser.add_argument("--folder-path", type=str)
    parser.add_argument("--pkl-path", type=str)
    args = parser.parse_args()
    eval_main(args)

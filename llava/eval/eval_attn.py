import re
import argparse
import torch
import requests
from io import BytesIO
from PIL import Image
import json
import pickle
from tqdm import tqdm
import time
import os
import signal

from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.model.builder import load_pretrained_model
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates

def load_json(data, folder_path):
    queries = []
    img_paths = []
    for i in data:
        queries.append(i['query'])
        img_paths.append(folder_path + "/" + i['imgname'])

    return queries, img_paths


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return [image]
    
def get_prompt(qs, model_name):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt

def eval_model(args):

    with open(args.json_path, 'r') as file:
        data = json.load(file)

    queries, img_paths = load_json(data, args.folder_path)

    final = []
    
    for i in tqdm(range(len(queries))):

        disable_torch_init()
        print("new code..")

        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, args.model_base, model_name, device_map="auto"
        )

        print("loaded model..")

        qs = queries[i]
        print(qs)
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        # if IMAGE_PLACEHOLDER in qs:
        #     if model.config.mm_use_im_start_end:
        #         qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        #     else:
        #         qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        # else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        prompt = get_prompt(qs, model_name)
        print(img_paths[i])
        images = load_image(img_paths[i])
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=512,
                use_cache=False,
                return_dict_in_generate=True,
                output_attentions=True
            )


        outputs = tokenizer.batch_decode(output_ids['sequences'], skip_special_tokens=True)[0].strip()

        final.append((i, output_ids['attentions'], output_ids['sequences'], outputs))
        #final.append((i, outputs))

        #del input_ids, images_tensor, output_ids, image_sizes
        #gc.collect()
        #torch.cuda.empty_cache()
        print(output_ids['sequences'])
        print("alright!")
        pid = os.getpid()
        os.kill(pid, signal.SIGTERM)


    with open(args.pkl_path, 'wb') as f:
        pickle.dump(data, f)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="lmms-lab/llama3-llava-next-8b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--json-path", type=str)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--folder-path", type=str)
    parser.add_argument("--pkl-path", type=str)
    args = parser.parse_args()
    eval_model(args)
# Visual Prompting Through Image-Mines [![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) [![jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626.svg?style=flat&logo=Jupyter)](https://jupyterlab.readthedocs.io/en/stable)

This repository contains all the codes for the paper, _Visual Prompting Through Image Mines_ submitted at [ICIP 2025](https://2025.ieeeicip.org/)

## Abstract
Visual prompting aims to enhance the performance of vision-language models (VLMs), which, despite their remarkable capabilities, often struggle with dense, detailed images, leading to incorrect answers or hallucinations. We propose _Visual Prompting Through Image Mines_, a novel algorithm that leverages attention patterns from a base VLM to generate image crops for improved visual grounding. Specifically, we extract attention values from output text tokens in the LLaVA-8B model, overlay them onto image patches to create an attention graph, and apply a modified breadth-first search (BFS) to identify key regions as image crops. Using the SigLIP model, we refine these regions into _Image Mines_, retaining only the most relevant crops. Our approach supports both single-image and multi-image inference setups, demonstrating superior performance compared to existing visual prompting methods.

## Overview
![Overview](pipeline.png)

## Getting Started
You can follow the instructions below to setup the code on your local system<br>

1. Clone this repository
```bash
git clone https://github.com/Snehan2k2/Image-Mines
cd Image-Mines
```
2. Create a virtual environment
```Shell
python -m venv image_mines_env
source image_mines_env/bin/activate
pip install --upgrade pip  # enable PEP 660 support
pip install -r requirements.txt
```
3. Loading the model
You can download the model(s) from huggingface.  
 - The base VLM weights are available at https://huggingface.co/lmms-lab/llama3-llava-next-8b
 - The weights of the SigLIP model are available at https://huggingface.co/google/siglip-so400m-patch14-384 
```Shell
huggingface-cli download lmms-lab/llama3-llava-next-8b
huggingface-cli download google/siglip-so400m-patch14-384
```

## Using the Algorithm
You can follow the steps below to generate attention maps and bounding boxes for a single image. The process is split into two main stages, each explained in a seperate notebook file<br>

1. **Generate attention map**

The first step involved generating an attention map for a single image, which highlights areas of interest based on the model's attention. The code for this step is explained in the [attn.ipynb](./attn.ipynb) file.

2. **Generate bounding boxes**

After obtaining the attention map, the next step is to generate bounding boxes that highlight the regions of interest. The corresponding code is explained in the [demo.ipynb](./demo.ipynb) file.

## Contributors
- Kalash Shah [![GitHub](https://i.sstatic.net/tskMh.png)](https://github.com/Kalash1106) [![Linkedin](https://i.sstatic.net/gVE0j.png)](https://www.linkedin.com/in/kalash-shah-b4567a20b)
- Snehan J [![GitHub](https://i.sstatic.net/tskMh.png)](https://github.com/Snehan2k2) [![Linkedin](https://i.sstatic.net/gVE0j.png)](https://www.linkedin.com/in/snehan-jayakumar-641964188)
- Gautam Bhutani
- Kunal Singh
- Shreyas Singh

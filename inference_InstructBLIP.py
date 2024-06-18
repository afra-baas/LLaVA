from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

import PIL
from PIL import Image, ImageDraw, ImageFont
import math
from io import BytesIO
import requests

import shutil
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import numpy as np

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
from llava.conversation import conv_templates


if __name__ == "__main__":

    # Load model and processor
    model_name = "Salesforce/instructblip-vicuna-7b"
    model = InstructBlipForConditionalGeneration.from_pretrained(model_name)
    processor = InstructBlipProcessor.from_pretrained(model_name)

    # Check available device and move model to device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    url = "/project/117.jpg"
    image = Image.open(url).convert("RGB")
    prompt = "What signs do you see in the image?"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device=device, dtype=torch.float16)

    # autoregressively generate an answer
    outputs = model.generate(
            **inputs,
            num_beams=5,
            max_new_tokens=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
    )
    outputs[outputs == 0] = 2 # this line can be removed once https://github.com/huggingface/transformers/pull/24492 is fixed
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print(generated_text)


#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
# from llava.model import *
from llava.model.language_model.llava_llama_dino import LlavaLlamaForCausalLM, LlavaConfig

from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.train.train_custom_dino import *

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    
    Freeze_VLM=False
    # Freeze_dino=True
    
    kwargs = {"device_map": device_map}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # added for dino cant be device_map =auto

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            print('lora in model_name.lower() and model_base is not None')
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = DepthLlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            model.initialize_weights()
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')

            # Remove LoRA renaming of parameters    
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            print("non_lora_trainables keys", non_lora_trainables.keys())
            model.load_state_dict(non_lora_trainables, strict=False)

            if Freeze_VLM == False:
                from peft import PeftModel
                print('Loading LoRA weights...')
                model = PeftModel.from_pretrained(model, model_path)
                print('Merging LoRA weights...')
                model = model.merge_and_unload()
                print('Model is loaded...')

            # if hasattr(model, "dino_model") and hasattr(model, "dino_feature_extractor"):
            #     dino_model_weight_path = os.path.join(model_path, 'dino_model.pth')
            #     dino_feature_extractor_weight_path = os.path.join(model_path, 'dino_feature_extractor.pth')

            #     if os.path.exists(dino_model_weight_path) and os.path.exists(dino_feature_extractor_weight_path):
            #         dino_model_state_dict = torch.load(dino_model_weight_path)
            #         dino_feature_extractor_state_dict = torch.load(dino_feature_extractor_weight_path)

            #         print('Loading DINO model weights...')
            #         model.dino_model.load_state_dict(dino_model_state_dict)

            #         print('Loading DINO feature extractor weights...')
            #         model.dino_feature_extractor.load_state_dict(dino_feature_extractor_state_dict)
            #     else:
            #         print("No weight paths found for DINO model or feature extractor")
            # else:
            #     print('Model does not have the attributes dino_model or dino_feature_extractor')

            if hasattr(model, "dino_model") :
                weight_path = os.path.join(model_path, 'dino_model.pth')
                if os.path.exists(weight_path):
                    state_dict = torch.load(weight_path)
                    # print(state_dict.items())
                    state_dict = {(k.replace('base_layer.', '') if 'base_layer.' in k else k): v for k, v in state_dict.items() if 'lora_' not in k}
                    print('Loading DINO model weights...')
                    # print(model.dino_model)
                    # print(state_dict.items())
                    model.dino_model.load_state_dict(state_dict)
                    # print(model.dino_model.state_dict())
                else:
                    print("No weight paths found for DINO model")
            else:
                print('Model does not have the attributes dino_model')

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len

import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
import os 
from datasets import load_dataset
from tqdm.auto import tqdm
from PIL import Image
import json
from io import BytesIO
import requests

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
from llava.conversation import conv_templates


def VSR_TF_prompt(batch_cap):
    return f"Given the image is the following statement True or False?\n{batch_cap}"

def pick_from_two_captions_prompt(batch_cap):
    # return f"Which of these 2 relations best describes the relation in the given image:\nA: {batch_cap[0]}\nB: {batch_cap[1]}\n"
    # return f"Identify the relation that best describes the image:\nA: {batch_cap}\nB: {batch_cap}\nChoose the most suitable option."
    return f"Identify the correct relation between the objects mentioned, given the image:\nA: {batch_cap[0]}\nB: {batch_cap[1]}\nChoose the most suitable option.\n"

def pick_from_four_captions_prompt(batch_cap):
    # return f"Which of these 4 relations best describes the given image:\nA: {batch_cap[0]}\nB: {batch_cap[1]}\nC: {batch_cap[2]}\nD: {batch_cap[3]}\n"
    return f"Identify the correct relation between the objects mentioned, given the image:\nA: {batch_cap[0]}\nB: {batch_cap[1]}\nC: {batch_cap[2]}\nD: {batch_cap[3]}\nChoose the most suitable option.\n"


class VSR_TF_Dataset(Dataset):
    def __init__(self, hf_dataset, processor, split):
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.split= split
        self.root_folder = '../../vsr_depth'
        # self.root_folder = '/project/msc-thesis-project/vsr_depth'

    def __len__(self):
        # return len(self.hf_dataset)
        return min(len(self.hf_dataset), 100)

    def __getitem__(self, idx):
        # Define how to retrieve a sample
        line=self.hf_dataset[idx]
        question = line['conversations'][0]['value']
        answer = line['conversations'][1]['value']
        qs = question.replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            if image_file.startswith("http") or image_file.startswith("https"):
                response = requests.get(image_file)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(os.path.join(image_file))

            # if getattr(model.config, 'mm_use_im_start_end', False):
            #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            # else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            print('It doesnt come here unless', line)

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # max_length =2048
        max_length =512
        encodings = self.processor(images = image, text = prompt, padding = "max_length",max_length=max_length, return_tensors = "pt").to(device, torch.float16)
        labels = self.processor(text = answer, padding = "max_length",max_length=max_length, return_tensors = "pt").input_ids

        encodings['labels'] = labels
        encodings = {k:v.squeeze() for k,v in encodings.items()}

        return encodings


def get_VSR_dataloader(processor, TF_task, split):
    dataset_name='VSR'
    batch_size = 4
    if TF_task:
        task='TF'
        # dataset = json.load(open(os.path.expanduser(f"./playground/data/eval/custom2/{dataset_name}_{split}_{task}.json"), "r"))
        # dataset = load_dataset("json", data_files=f"/project/msc-thesis-project/LLaVA/playground/data/eval/custom2/{dataset_name}_{split}_{task}.json", field="validation")["train"]
        dataset = load_dataset("json", data_files={'train':f"/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/{dataset_name}_train_{task}.json", 
                                         'val':f"/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/{dataset_name}_val_{task}.json", 
                                         'test':f"/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/{dataset_name}_test_{task}.json"})

        data_loader = DataLoader(VSR_TF_Dataset(dataset[split], processor, f'{split}'), batch_size=batch_size, shuffle=True)
    return data_loader


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def train():
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj"]
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.to(device)

    data_loader = get_VSR_dataloader(processor, TF_task= True, split='val')

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    model.train()
    for epoch in range(1):
        print("Epoch:", epoch)
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for idx, batch in pbar: 
            input_ids =batch
            
            outputs = model(**input_ids)

            loss = outputs.loss
            print("Loss:", loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Save checkpoint
            if idx % save_interval == 0:
                model.save_pretrained(f"./checkpoints/{CKPT}/epoch_{epoch}_batch_{idx}")
                model.config.save_pretrained(f"./checkpoints/{CKPT}/epoch_{epoch}_batch_{idx}")

            # if lora_enable:
            #     state_dict = get_peft_state_maybe_zero_3(
            #         model.named_parameters(), training_args.lora_bias
            #     )
            #     non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            #         model.named_parameters()
            #     )
            #     if training_args.local_rank == 0 or training_args.local_rank == -1:
            #         model.config.save_pretrained(training_args.output_dir)
            #         model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            #         torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            # else:
            #     safe_save_model_for_hf_trainer(trainer=trainer,
            #                                 output_dir=training_args.output_dir)


# https://github.com/haotian-liu/LLaVA/issues/729
                
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conv_mode= "vicuna_v1"
    CKPT= "VSR_TF_epoch3-nodepth-blip2_second_try"
    save_interval = 100
    train()

# source: https://github.com/huggingface/notebooks/blob/main/peft/Fine_tune_BLIP2_on_an_image_captioning_dataset_PEFT.ipynb
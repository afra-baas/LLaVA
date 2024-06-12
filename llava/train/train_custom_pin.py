import os
from pathlib import Path

# from llava.model import *

from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

if __name__=="__main__":
    print('using monkey patch')
    replace_llama_attn_with_flash_attn()  # Need to call this before importing transformers.

from llava.model.language_model.llava_llama_pin import LlavaLlamaForCausalLM, AutoModelForCausalLM, LlavaConfig
from llava.train.train_for_pin import *
import torch.nn as nn
import torch.nn.init as init
from io import BytesIO
from transformers.generation.utils import GenerateOutput
from typing import List, Optional, Tuple, Union

import torch
import numpy as np

def create_positional_embeddings(N_p, d_model):
    S = np.zeros((N_p, d_model))
    for i in range(N_p):
        for k in range(d_model // 2):
            S[i, 2*k] = np.sin(i / (10000 ** (2 * k / d_model)))
            S[i, 2*k + 1] = np.cos(i / (10000 ** (2 * k / d_model)))
    return torch.tensor(S, dtype=torch.float32)

class PIN(nn.Module):
    def __init__(self, N_p, d_model):
        super(PIN, self).__init__()
        self.positional_embeddings = create_positional_embeddings(N_p, d_model)
        self.refinement_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self):
        refined_embeddings = self.refinement_net(self.positional_embeddings)
        return refined_embeddings


class DepthLlavaLlamaForCausalLM(LlavaLlamaForCausalLM):
    config_class = LlavaConfig
    torch.manual_seed(25)

    def __init__(self, config): #, device=None -> error: TypeError: Object of type device is not JSON serializable  
        super(DepthLlavaLlamaForCausalLM, self).__init__(config)
        print('DepthLlavaLlamaForCausalLM resnet')

        N_p = 50  # Number of patches
        D_v = 512  # Dimension of vision encoder output
        self.pin = PIN(N_p, D_v)
        
    def initialize_weights(self):
        print("CHECK: initialize_weights")

        # print('printing showed that parameters are initialzied for training')

        # for name, param in self.depth_cnn.named_parameters():
        #     print('depth_cnn')
        #     if 'weight' in name:
        #         print(f'Parameter: {name}')
        #         print(f'Initial weights:')
        #         print(param.data)

        for name, param in self.depth_cnn.named_parameters(): # all have dtype bfloat16
            if name=='cnn.0.weight':
                print(f"Parameter {name}: dtype={param.dtype}")
                print('dtype init weights !!!')
                break
            

        # for name, param in self.fusion_module.named_parameters():
        #     print('fusion_module')
        #     print(f"Parameter {name}: dtype={param.dtype}")


        # nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.zeros_(self.fc1.bias)

        # nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.zeros_(self.fc2.bias)



    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        depth_images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ):

        if inputs_embeds is None:
            (input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                depth_images
            )
        

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        depth_images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        # print("kwargs", kwargs)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        print("DepthLlavaLlamaForCausalLM generate resnet ")
        # print("MRO:", [cls.__name__ for cls in DepthLlavaLlamaForCausalLM.mro()])
        
        return super().generate(
            inputs,
            images,
            depth_images,
            image_sizes=image_sizes,
            **kwargs,
        )

    
    def encode_images(self, images, depth_images=None):
        return super().encode_images(images, depth_images=depth_images)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        print("prepare_inputs_for_generation resnet ")
        images = kwargs.pop("images", None)
        depth_images = kwargs.pop("depth_images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        
        if depth_images is not None:
            _inputs['depth_images'] = depth_images
        return _inputs

AutoModelForCausalLM.register(LlavaConfig, DepthLlavaLlamaForCausalLM)


################## Dataset ##################

class DepthSupervisedDataset(LazySupervisedDataset):
    def __init__(self, *args, depth_path='', **kwargs):
        super(DepthSupervisedDataset, self).__init__(*args, **kwargs)
        self.depth_path = depth_path

    def __getitem__(self, i):
        data_dict = super().__getitem__(i)
        if 'image' in self.list_data_dict[i] and isinstance(self.list_data_dict[i]['image'], str):
            image_file = Path(self.list_data_dict[i]['image']).name
            image_folder = self.depth_path
            processor = self.data_args.image_processor

            if image_file.startswith("http") or image_file.startswith("https"):
                response = requests.get(image_file)
                depth_image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                depth_image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')

            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                depth_image = expand2square(depth_image, tuple(int(x*255) for x in processor.image_mean))
                depth_image = processor.preprocess(depth_image, return_tensors='pt')['pixel_values'][0]
            else:
                depth_image = processor.preprocess(depth_image, return_tensors='pt')['pixel_values'][0]

            #processor.preprocess converts image into tensor of shape [3,336,336]
            data_dict['depth_image'] = depth_image

        return data_dict


@dataclass
class DataCollatorForDepthSupervisedDataset(DataCollatorForSupervisedDataset):
    def __call__(self, instances):
        batch = super().__call__(instances)
        if 'depth_image' in instances[0]:
            depth_images = [instance['depth_image'] for instance in instances]
            if all(x is not None and x.shape == depth_images[0].shape for x in depth_images):
                batch['depth_images'] = torch.stack(depth_images)
            else:
                batch['depth_images'] = depth_images

        return batch


def custom_make_supervised_data_module(tokenizer, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = DepthSupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                depth_path=data_args.depth_path_train)
    
    eval_dataset = DepthSupervisedDataset(tokenizer=tokenizer,
                            data_path=data_args.validation_data_path,
                            data_args=data_args,
                            depth_path=data_args.depth_path_val)  
    
    data_collator = DataCollatorForDepthSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


################## Train Code ##################

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    # Model init
    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = DepthLlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                # low_cpu_mem_usage=True, # ValueError: DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True` or with passing a `device_map`.
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
            model.initialize_weights()
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # print('state_dict.keys() before lora ', model.fusion_layer.state_dict().keys()) # (['weight', 'bias'])
    # LORA adds adapters, used to adaptively adjust the learning rate of each weight in the layer based on its importance for the task at hand.
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # print('state_dict.keys() after lora ', model.fusion_layer.state_dict().keys()) # (['base_layer.weight', 'base_layer.bias', 'lora_A.default.weight', 'lora_B.default.weight'])  
    
    # Tokenizer init
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        
        # check if projection is being trained -> base_model.model.model.mm_projector -> all weights and bias are true
        model_name = model.__class__.__name__
        print(f"Model name: {model_name}")
        for name, param in model.named_parameters():
            print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
        # model_model_name = model.model.__class__.__name__ # DepthLlavaLlamaForCausalLM


        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # Data init, replaced with custom dataset for including depth maps
    data_module = custom_make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    print('list checkpoint', list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))) #[]
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # torch.cuda.empty_cache()
    model.config.use_cache = True


    # Check and save state_dict for DepthCNN
    depth_cnn_state_dict = model.depth_cnn.state_dict()
    for name, param in depth_cnn_state_dict.items():
        if 'weight' in name:
            weight_has_issue = torch.isnan(param).any() or torch.isinf(param).any()
            print("DepthCNN - Weight has NaN, -inf, or inf:", weight_has_issue)
        elif 'bias' in name:
            bias_has_issue = torch.isnan(param).any() or torch.isinf(param).any()
            print("DepthCNN - Bias has NaN, -inf, or inf:", bias_has_issue)
    torch.save(depth_cnn_state_dict, os.path.join(training_args.output_dir, 'depth_cnn.pth'))

    # Check and save state_dict for FusionModule
    fusion_state_dict = model.fusion_module.state_dict()
    weight_has_issue = torch.isnan(fusion_state_dict['fc1.weight']).any() or torch.isinf(fusion_state_dict['fc1.weight']).any()
    bias_has_issue = torch.isnan(fusion_state_dict['fc1.bias']).any() or torch.isinf(fusion_state_dict['fc1.bias']).any()
    print("FusionModule - Weight has NaN, -inf, or inf:", weight_has_issue)
    print("FusionModule - Bias has NaN, -inf, or inf:", bias_has_issue)
    torch.save(fusion_state_dict, 'fusion_module.pth')


    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

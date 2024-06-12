import os
from pathlib import Path

from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

if __name__=="__main__":
    print('using monkey patch')
    replace_llama_attn_with_flash_attn()  # Need to call this before importing transformers.

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, AutoModelForCausalLM, LlavaConfig
from llava.train.train import *
import torch.nn as nn
import torch.nn.init as init

from transformers.generation.utils import GenerateOutput
from typing import List, Optional, Tuple, Union

class DepthLlavaLlamaForCausalLM(LlavaLlamaForCausalLM):
    config_class = LlavaConfig
    torch.manual_seed(25)

    def __init__(self, config):
        super(DepthLlavaLlamaForCausalLM, self).__init__(config)
        self.conv1x1 = nn.Conv2d(4, 3, kernel_size=1)#.half()
        # self.conv1x1.requires_grad_(True)
        # self.conv1x1 = self.conv1x1.to(self.device)
        self.conv_weights_path = 'conv1x1_weights.pth'
        print('state_dict conv init', self.conv1x1.state_dict()) 

    # TODO: change to _init_weights everywhere
    def initialize_weights(self):
        print("CHECK: initialize_weights")
        nn.init.xavier_uniform_(self.conv1x1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.conv1x1.bias)
        print('state_dict conv initialize_weights', self.conv1x1.state_dict()) 
        file = "/project/train_custom_conv_state_dict.txt"
        with open(file, "w") as f:
            f.write(f'state_dict conv after init weights: \n{self.conv1x1.state_dict()}')


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
            if images is not None and depth_images is not None:
                # first convert depth to grayscale
                depth_images = depth_images.mean(dim=1, keepdim=True)
                images = torch.cat((images, depth_images), dim=1) #torch.Size([16, 4, 336, 336]) -> torch.Size([16, 3, 336, 336])
                # images = torch.cat((images, depth_images), dim=-1)  # dim 1 or -1? #torch.Size([16, 3, 336, 672])

            (
                input_ids,
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
                images
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
        
        # print("DepthLlavaLlamaForCausalLM generate")
        # print("MRO:", [cls.__name__ for cls in DepthLlavaLlamaForCausalLM.mro()])
        
        if images is not None:
            depth_images = depth_images.mean(dim=1, keepdim=True)
            images = torch.cat((images, depth_images), dim=1) #torch.Size([16, 4, 336, 336]) -> torch.Size([16, 3, 336, 336])
            # images = torch.cat((images, depth_images), dim=-1)  # dim 1 or -1? #torch.Size([16, 3, 336, 672])
            # print("image and depth map concatenated before encoder in generate")

        return super().generate(
            inputs,
            images,
            image_sizes=image_sizes,
            **kwargs,
        )
    
    def encode_images(self, images):
        # print(type(self.conv1x1)) #<class 'torch.nn.modules.conv.Conv2d'>  
        # print(type(images)) #<class 'torch.Tensor'>
        images = self.conv1x1(images)
        # print("(images.shape na conv1x1", images.shape)
        return super().encode_images(images)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # print("prepare_inputs_for_generation")
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
            image_name = Path(self.list_data_dict[i]['image']).name
            depth_folder = self.depth_path
            depth_image = Image.open(os.path.join(depth_folder, image_name)).convert('RGB')
            processor = self.data_args.image_processor

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
    # depth_path='/project/msc-thesis-project/vsr_depth/val/'

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
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = DepthLlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
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
        print('model_args.freeze_backbone')
        model.model.requires_grad_(False)

    # so llm is frozen but then later lora unfreezes it 
    # model_name = model.__class__.__name__
    # file = "/project/train_custom_conv_all_requires_grad_should_be_only_conv_test.txt"
    # with open(file, "w") as f:
    #     f.write(f"Model name: {model_name}\n")
    #     for name, param in model.named_parameters():
    #         # if param.requires_grad:
    #         f.write(f"Parameter name: {name}, requires_grad: {param.requires_grad}\n")

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

    # model_name = model.__class__.__name__
    # file = "/project/train_custom_conv_all_requires_grad_should_be_only_conv_test.txt"
    # with open(file, "w") as f:
    #     f.write(f"Model name: {model_name}\n")
    #     for name, param in model.named_parameters():
    #         # if param.requires_grad:
    #         f.write(f"Parameter name: {name}, requires_grad: {param.requires_grad}\n")

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
        # model = get_peft_model(model, lora_config)

    # I added 
    model.lm_head.weight.requires_grad = False
    # print(model.lm_head.weight)

    # model_name = model.__class__.__name__
    # file = "/project/train_custom_conv_all_requires_grad_should_be_only_conv_test2.txt"
    # with open(file, "w") as f:
    #     f.write(f"Model name: {model_name}\n")
    #     for name, param in model.named_parameters():
    #         # if param.requires_grad:
    #         f.write(f"Parameter name: {name}, requires_grad: {param.requires_grad}\n")

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
            print('Let op!!! model.requires_grad_(False)')
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter: # what is the mm_mlp_adapter?
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.conv1x1.requires_grad_(True)

        from peft.tuners.lora import LoraLayer
        model_name = model.__class__.__name__
        file = "/project/train_custom_conv_all_requires_grad_should_be_only_conv.txt"
        with open(file, "w") as f:
            f.write(f"Model name: {model_name}\n")
            for name, param in model.named_parameters():
                # if param.requires_grad:
                f.write(f"Parameter name: {name}, requires_grad: {param.requires_grad}\n")
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"Trainable parameters: {trainable_params}\n")
            total_params = sum(p.numel() for p in model.parameters())
            f.write(f"Total parameters: {total_params}\n")

            trainable_lora_params=0
            total_lora_params=0
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    for param_name, param in module.named_parameters():
                        if param.requires_grad:
                            trainable_lora_params += param.numel()
                        total_lora_params += param.numel()
            f.write(f"Trainable parameters in LoRA layers: {trainable_lora_params}\n")
            f.write(f"Total parameters in LoRA layers: {total_lora_params}\n")


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

    print('list checkpoint', list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")))
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    torch.cuda.empty_cache()
    model.config.use_cache = True

    conv1x1_state_dict = model.conv1x1.state_dict()
    # Check for NaNs, -inf, and inf 
    weight_has_issue = torch.isnan(conv1x1_state_dict['weight']).any() or torch.isinf(conv1x1_state_dict['weight']).any()
    bias_has_issue = torch.isnan(conv1x1_state_dict['bias']).any() or torch.isinf(conv1x1_state_dict['bias']).any()

    # Print the results
    print("Weight has NaN, -inf, or inf:", weight_has_issue)
    print("Bias has NaN, -inf, or inf:", bias_has_issue)

    print('state_dict conv before save', conv1x1_state_dict) 
    weight_path = os.path.join(training_args.output_dir, model.conv_weights_path)
    print('weight path', weight_path)
    torch.save(conv1x1_state_dict, weight_path)

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

    file = "/project/train_custom_conv_state_dict.txt"
    with open(file, "a") as f:
        f.write(f'state_dict conv before save: \n{conv1x1_state_dict}')


if __name__ == "__main__":
    train()

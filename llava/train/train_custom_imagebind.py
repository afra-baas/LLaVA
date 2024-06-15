import os
from pathlib import Path

import sys 
# Add the ImageBind folder to the Python path
sys.path.append("/project/msc-thesis-project/forked_repos/ImageBind")

# Add the LLaVA folder to the Python path
sys.path.append("/project/msc-thesis-project/forked_repos/LLaVA")


from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

if __name__=="__main__":
    print('using monkey patch')
    replace_llama_attn_with_flash_attn()  # Need to call this before importing transformers.

from llava.model.language_model.llava_llama_imagebind import LlavaLlamaForCausalLM, AutoModelForCausalLM, LlavaConfig
from llava.train.train_for_imagebind import *
import torch.nn as nn
import torch.nn.init as init

from transformers.generation.utils import GenerateOutput
from typing import List, Optional, Tuple, Union
from datetime import datetime
dt = datetime.now()


from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data

from torchvision import transforms
from PIL import Image
def load_and_transform_depth_data(depth_paths, device=None):
    if depth_paths is None:
        return None

    depth_ouputs = []
    for depth_path in depth_paths:
        data_transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, ), (0.5, ))  # if I use this normalization, I cannot get good results...
            ]
        )
        # data_transform = transforms.Compose(
        #     [
        #         transforms.Resize(
        #             336, interpolation=transforms.InterpolationMode.BICUBIC
        #         ),
        #         transforms.CenterCrop(336),
        #         transforms.ToTensor(),
        #         # transforms.Normalize((0.5, ), (0.5, ))  # if I use this normalization, I cannot get good results...
        #     ]
        # )
        with open(depth_path, "rb") as fopen:
            image = Image.open(fopen).convert("L")

        image = data_transform(image)#.to(device)
        depth_ouputs.append(image)
    return torch.stack(depth_ouputs, dim=0)

# Instantiate model
imagebindm = imagebind_model.imagebind_huge(pretrained=True)
imagebindm.eval()
# imagebindm.to(device)

class DepthLlavaLlamaForCausalLM(LlavaLlamaForCausalLM):
    config_class = LlavaConfig
    torch.manual_seed(25)

    def __init__(self, config):
        super(DepthLlavaLlamaForCausalLM, self).__init__(config)
        print(self.device)
        # device = "cuda:1" if torch.cuda.is_available() else "cpu"

        # # Instantiate model
        # self.imagebindm = imagebind_model.imagebind_huge(pretrained=True)
        # self.imagebindm.eval()
        # # self.imagebindm.to(device)


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
            # if images is not None and depth_images is not None:
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
        
        return super().generate(
            inputs,
            images,
            depth_images,
            image_sizes=image_sizes,
            **kwargs,
        )
    
    def encode_images(self, images, depth_images=None):
        global imagebindm
        # print('self.device', self.device)
        # images = images.to(self.device)

        clip_image_features= super().encode_images(images) #([16, 576, 4096]) 
        # image_features = self.get_model().get_vision_tower()(images) #([16, 576, 1024]) 

        depth_images= depth_images.to(self.device)
        # print(depth_images.shape) # (1,1,224,224)
        inputs = {
            # ModalityType.TEXT: data.load_and_transform_text(text_list, device),
            # ModalityType.VISION: data.load_and_transform_vision_data(image_paths),
            ModalityType.DEPTH: depth_images
        }

        imagebindm=imagebindm.half().to(self.device)
        # print('imagebindm',imagebindm.device)
        with torch.no_grad():
            embeddings = imagebindm(inputs)

        depth_features= embeddings[ModalityType.DEPTH] #([16, 1024]) 

        # print('clip_image_features.shape',clip_image_features.shape)
        # print('depth _features.shape',embeddings[ModalityType.DEPTH].shape)
        # print(embeddings[ModalityType.VISION].shape)

        # depth_features = self.get_model().mm_projector(depth_features).unsqueeze(1) #([16, 1, 4096]) 
        depth_features = self.get_model().linear_depth_projector(depth_features).unsqueeze(1) #([16, 1, 4096]) 
        # return clip_image_features,depth_features

        print('depth_features shape ', depth_features.shape)
        image_features = torch.cat((clip_image_features, depth_features), dim=1)
        print('image_features shape ', image_features.shape)
        return image_features
    
    # def prepare_inputs_labels_for_multimodal(
    #     self, input_ids, position_ids, attention_mask, past_key_values, labels, images, depth_images, image_sizes=None
    # ):
    #     print("prepare_inputs_labels_for_multimodal in LlavaMetaForCausalLM in imagebind late")
    #     vision_tower = self.get_vision_tower()
    #     if vision_tower is None or images is None or input_ids.shape[1] == 1:
    #         if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
    #             print(" hierr extra check in late")
    #         return input_ids, position_ids, attention_mask, past_key_values, None, labels

    #     if type(images) is list or images.ndim == 5:
    #         if type(images) is list:
    #             print('need extra check from arch in late')
    #         print('--- Let op images are a list')

    #         concat_images_depth = torch.cat([image for image in depth_images], dim=0)
    #         concat_images = torch.cat([image for image in images], dim=0)
    #         image_features = self.encode_images(concat_images,concat_images_depth)
    #         split_sizes = [image.shape[0] for image in images]
    #         image_features,depth_image_features = torch.split(image_features, split_sizes, dim=0)
            
    #         # image_features = [x.flatten(0, 1).to(concat_images.device) for x in image_features]
    #         mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
    #         image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
    #         if mm_patch_merge_type == 'flat':
    #             image_features = [x.flatten(0, 1) for x in image_features]
    #         elif mm_patch_merge_type.startswith('spatial'):
    #             new_image_features = []
    #             for image_idx, image_feature in enumerate(image_features):
    #                 if image_feature.shape[0] > 1:
    #                     base_image_feature = image_feature[0]
    #                     image_feature = image_feature[1:]
    #                     height = width = self.get_vision_tower().num_patches_per_side
    #                     assert height * width == base_image_feature.shape[0]
    #                     if image_aspect_ratio == 'anyres':
    #                         num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
    #                         image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
    #                     else:
    #                         raise NotImplementedError
    #                     if 'unpad' in mm_patch_merge_type:
    #                         image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
    #                         image_feature = image_feature.flatten(1, 2).flatten(2, 3)
    #                         print("image_sizes is being used")
    #                         image_feature = unpad_image(image_feature, image_sizes[image_idx])
    #                         image_feature = torch.cat((
    #                             image_feature,
    #                             self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
    #                         ), dim=-1)
    #                         image_feature = image_feature.flatten(1, 2).transpose(0, 1)
    #                     else:
    #                         image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
    #                         image_feature = image_feature.flatten(0, 3)
    #                     image_feature = torch.cat((base_image_feature, image_feature), dim=0)
    #                 else:
    #                     image_feature = image_feature[0]
    #                     if 'unpad' in mm_patch_merge_type:
    #                         image_feature = torch.cat((
    #                             image_feature,
    #                             self.model.image_newline[None].to(image_feature.device)
    #                         ), dim=0)
    #                 new_image_features.append(image_feature)
    #             image_features = new_image_features
    #         else:
    #             raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")

    #     else:
    #         image_features,depth_image_features = self.encode_images(images,depth_images)


    #     # TODO: image start / end is not implemented here to support pretraining.
    #     if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
    #         raise NotImplementedError

    #     # Let's just add dummy tensors if they do not exist,
    #     # it is a headache to deal with None all the time.
    #     # But it is not ideal, and if you have a better idea,
    #     # please open an issue / submit a PR, thanks.
    #     _labels = labels
    #     _position_ids = position_ids
    #     _attention_mask = attention_mask
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    #     else:
    #         attention_mask = attention_mask.bool()
    #     if position_ids is None:
    #         position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    #     if labels is None:
    #         labels = torch.full_like(input_ids, IGNORE_INDEX)

    #     # remove the padding using attention_mask -- TODO: double check
    #     input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    #     labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    #     new_input_embeds = []
    #     new_labels = []
    #     cur_image_idx = 0
    #     for batch_idx, cur_input_ids in enumerate(input_ids):
    #         # print('cur_input_ids', cur_input_ids.shape) #[68]
    #         num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
    #         # print('num_images', num_images)
    #         if num_images == 0:
    #             cur_image_features = image_features[cur_image_idx]
    #             cur_depth_image_features = depth_image_features[cur_image_idx]
    #             cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
    #             # i added because of not same device error
    #             cur_image_features= cur_image_features.to(self.device) # torch.Size([576, 4096]) 
    #             cur_depth_image_features= cur_depth_image_features.to(self.device) # torch.Size([..., 4096]) 
    #             cur_input_embeds_1= cur_input_embeds_1.to(self.device)# torch.Size([100, 4096]) 
    #             print('cur_input_embeds_1', cur_input_embeds_1.shape ) 
    #             print('cur_image_features', cur_image_features.shape)
    #             print('cur_depth_image_features', cur_depth_image_features.shape)
    #             # print(cur_image_features[0:0].shape) # torch.Size([0, 4096]) 
    #             cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_depth_image_features[0,0]], dim=0)
    #             print("cur_input_embeds", cur_input_embeds.shape) # torch.Size([100, 4096]) 
    #             # print("new_labels[i]", labels[batch_idx].shape) # torch.Size([100]) 
    #             new_input_embeds.append(cur_input_embeds)
    #             new_labels.append(labels[batch_idx])
    #             cur_image_idx += 1
    #             continue

    #         # print('---------------where IMAGE_TOKEN_INDEX ', torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist())  # idx of -200 is e.g. 35
    #         # print('num_images', num_images) #1
    #         # print(cur_input_ids)
    #         # print(IMAGE_TOKEN_INDEX) -200

    #         image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
    #         # print(image_token_indices) # [-1, 35, 142]
    #         cur_input_ids_noim = []
    #         cur_labels = labels[batch_idx]
    #         cur_labels_noim = []
    #         for i in range(len(image_token_indices) - 1):
    #             cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
    #             cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
    #         split_sizes = [x.shape[0] for x in cur_labels_noim]
    #         cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
    #         # print('cur_input_embeds', cur_input_embeds.shape) # [67, 4096], because 35 removed
    #         cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
    #         # print('cur_input_embeds_no_im', len(cur_input_embeds_no_im)) #2
    #         # print('cur_input_embeds_no_im', cur_input_embeds_no_im[0].shape) # [35,4096] and [1] = [the rest, 4096]

    #         cur_new_input_embeds = []
    #         cur_new_labels = []
    #         for i in range(num_images + 1):
    #             cur_new_input_embeds.append(cur_input_embeds_no_im[i])
    #             cur_new_labels.append(cur_labels_noim[i])
    #             if i < num_images:
    #                 cur_image_features = image_features[cur_image_idx]
    #                 cur_depth_image_features = depth_image_features[cur_image_idx]
    #                 # print('cur_image_features', cur_image_features.shape)  # [576,4096]
    #                 # print('cur_depth_image_features', cur_depth_image_features.shape)  # [1,4096]
    #                 cur_image_idx += 1
    #                 cur_new_input_embeds.append(cur_image_features)
    #                 cur_new_input_embeds.append(cur_depth_image_features)
    #                 # cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
    #                 cur_new_labels.append(torch.full((cur_image_features.shape[0] + cur_depth_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

    #         # i added because of not same device error
    #         cur_new_input_embeds = [embed.to(cur_labels.device) for embed in cur_new_input_embeds]
    #         # print('cur_new_input_embeds', len(cur_new_input_embeds)) #3
    #         cur_new_input_embeds = torch.cat(cur_new_input_embeds)
    #         # print('cur_new_input_embeds', cur_new_input_embeds.shape) # [1219,4096]
    #         cur_new_labels = torch.cat(cur_new_labels)

    #         new_input_embeds.append(cur_new_input_embeds)
    #         new_labels.append(cur_new_labels)

    #     # print('new_input_embeds len ', len(new_input_embeds)) # [32, torch.Size([1219, 4096])] 

    #     # Truncate sequences to max length as image embeddings can make the sequence longer
    #     tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
    #     if tokenizer_model_max_length is not None:
    #         new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
    #         new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    #     # Combine them
    #     max_len = max(x.shape[0] for x in new_input_embeds)
    #     batch_size = len(new_input_embeds)
    #     # print('max len and batch size', max_len, batch_size) # 1222 32

    #     new_input_embeds_padded = []
    #     new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    #     attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    #     position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    #     for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
    #         cur_len = cur_new_embed.shape[0]
    #         if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
    #             new_input_embeds_padded.append(torch.cat((
    #                 torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
    #                 cur_new_embed
    #             ), dim=0))
    #             if cur_len > 0:
    #                 new_labels_padded[i, -cur_len:] = cur_new_labels
    #                 attention_mask[i, -cur_len:] = True
    #                 position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
    #         else:
    #             new_input_embeds_padded.append(torch.cat((
    #                 cur_new_embed,
    #                 torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
    #             ), dim=0))
    #             if cur_len > 0:
    #                 new_labels_padded[i, :cur_len] = cur_new_labels
    #                 attention_mask[i, :cur_len] = True
    #                 position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    #     # print('new_input_embeds_padded', len(new_input_embeds_padded)) #32
    #     new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
    #     # print('new_labels_padded', len(new_labels_padded)) #32

    #     if _labels is None:
    #         new_labels = None
    #     else:
    #         new_labels = new_labels_padded

    #     if _attention_mask is None:
    #         attention_mask = None
    #     else:
    #         attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    #     if _position_ids is None:
    #         position_ids = None

    #     return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, depth_images, image_sizes=None
    ):
        print("prepare_inputs_labels_for_multimodal in LlavaMetaForCausalLM in arch in imagebind custon")
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                print(" hierr extra check in arch method 2")
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                print('type(images) is list')
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            concat_images_depth = torch.cat([image for image in depth_images], dim=0)
            
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images, concat_images_depth)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            # image_features = [x.flatten(0, 1).to(concat_images.device) for x in image_features]

            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                print('mm_patch_merge_type is flat')
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                print('mm_patch_merge_type is spatial')
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            print("image_sizes is being used")
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images, depth_images)
        
        # print("image_features in arch ", image_features.shape) #[1, 576, 4096]
        
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # print('cur_input_ids', cur_input_ids.shape)
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                # print('cur_input_embeds_1', cur_input_embeds_1.shape ) 
                # print('cur_image_features', cur_image_features.shape)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                # print("cur_input_embeds", cur_input_embeds.shape)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            # print('cur_input_embeds', cur_input_embeds.shape)
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    # print('num_images', num_images) 
                    # print('cur_image_idx', cur_image_idx) 
                    # print('image_features', image_features.shape)  #image_features torch.Size([8, 576, 4096])  
                    cur_image_features = image_features[cur_image_idx]
                    # print('cur_image_features', cur_image_features.shape) 
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            # print('cur_new_input_embeds', cur_new_input_embeds.shape)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

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


            # instead of preprocessing for llava's clip VE
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
            # print('type depth_image', type(depth_image))
            # print('depth_image shape ', depth_image.shape)



        
            depth_path = os.path.join(depth_folder, image_name)
            # print('depth_path', depth_path)
            depth_image= load_and_transform_depth_data([depth_path])[0]
            # print('type depth_image============2 ', type(depth_image))
            # print(depth_image.shape)

            # image_path = self.list_data_dict[i]['image']
            # img= data.load_and_transform_vision_data([image_path], 'cuda:0')[0]
            # print(type(data_dict['image']))
            # print('data_dict[image].shape',data_dict['image'].shape)
            # print(type(img))
            # print('img.shape',img.shape)
            
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
        print('hier bij fill in bnb_model_from_pretrained_args')
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

    # added freezebackbone
    model.model.requires_grad_(False)
    

    # so llm is frozen but then later lora unfreezes it 
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

    # I added for when you want to freeze everything
    model.lm_head.weight.requires_grad = False
    # print(model.lm_head.weight)

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

        # i added freeze proj 
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

        for p in model.get_model().linear_depth_projector.parameters():
            p.requires_grad = True

        for p in imagebindm.parameters():
            p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        from peft.tuners.lora import LoraLayer
        model_name = model.__class__.__name__
        file = f"/project/model_states/train_custom_imagebind_all_requires_grad_{dt}.txt"
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

        file = f"/project/model_states/train_imagebind_linear_depth_proj_{dt}.txt"
        with open(file, "w") as f:
            for name, param in model.get_model().linear_depth_projector.named_parameters():
                f.write(f"Parameter: {name}\n")
                # f.write(f"{param}\n")
                f.write(str(param.cpu().detach().to(torch.float32).numpy()))
                f.write("----------------------------------------------")
                f.write("\n")

    # file = f"/project/model_states/train_imagebind_mm_proj_{dt}.txt"
    # with open(file, "a") as f:
    #     f.write("=====================after===============================")
    #     for name, param in model.get_model().mm_projector.named_parameters():
    #         f.write(f"Parameter: {name}\n")
    #         # f.write(f"{param}\n")
    #         f.write(str(param.cpu().detach().to(torch.float32).numpy()))
    #         f.write("----------------------------------------------")
    #         f.write("\n")

        file = f"/project/model_states/train_custom_imagebind_params_{dt}.txt"
        with open(file, "w") as f:
            for name, param in imagebindm.named_parameters():
                # if param.requires_grad:
                f.write(f"Parameter name: {name}, requires_grad: {param.requires_grad}\n")
            trainable_params = sum(
                p.numel() for p in imagebindm.parameters() if p.requires_grad)
            f.write(f"Trainable parameters: {trainable_params}\n")
            total_params = sum(p.numel() for p in imagebindm.parameters())
            f.write(f"Total parameters: {total_params}\n")

            # for name, param in imagebindm.named_parameters():
            #     f.write(f"Parameter: {name}\n")
            #     # f.write(f"{param}\n")
            #     f.write(str(param.cpu().detach().to(torch.float32).numpy()))
            #     f.write("----------------------------------------------")
            #     f.write("\n")

            

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


    file = f"/project/model_states/train_imagebind_linear_depth_proj_{dt}.txt"
    with open(file, "a") as f:
        f.write("=====================after===============================")
        for name, param in model.get_model().linear_depth_projector.named_parameters():
            f.write(f"Parameter: {name}\n")
            # f.write(f"{param}\n")
            f.write(str(param.cpu().detach().to(torch.float32).numpy()))
            f.write("----------------------------------------------")
            f.write("\n")

    file = f"/project/model_states/train_custom_imagebind_params_{dt}.txt"
    with open(file, "a") as f:
        for name, param in imagebindm.named_parameters():
            f.write(f"Parameter: {name}\n")
            # f.write(f"{param}\n")
            f.write(str(param.cpu().detach().to(torch.float32).numpy()))
            f.write("----------------------------------------------")
            f.write("\n")


if __name__ == "__main__":
    train()

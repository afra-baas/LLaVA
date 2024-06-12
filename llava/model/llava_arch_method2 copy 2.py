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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def encode_images(self, images, depth_images=None):
        image_features = self.get_model().get_vision_tower()(images)
        print("image_features na alleen vision encoder", image_features.shape)# (32, 576, 1024)

        # # Resize the depth map
        # # print('depth_images', depth_images.shape)
        # resized_depth_map = nn.functional.interpolate(depth_images, size=(576, 1024), mode='bilinear')
        # # print("resized_depth_map", resized_depth_map.shape)
        # resized_depth_map = torch.mean(resized_depth_map, dim=1, keepdim=True).squeeze(1)
        # # print("resized_depth_map", resized_depth_map.shape)
        # # print("resized_depth_map", resized_depth_map)
        # # print(image_features)
        # # image_features =image_features +resized_depth_map
        # # print(image_features)
        # # image_features =image_features *resized_depth_map
        # # print(image_features)


        # # Method Conv for downsampling
        # desired_output_shape = (576, 1024)
        # input_shape = (336, 336)
        # input_channels = 3

        # # stride = (0.5833, 0.3281)
        # stride = (1, 1)
        # padding = (71, 155)

        # # Calculate the stride required to achieve the desired output shape
        # # stride = (input_shape[0] // desired_output_shape[0], input_shape[1] // desired_output_shape[1])
        # print('stride', stride)

        # # Calculate the padding required to achieve the desired output shape
        # # Padding ensures that the input size matches the desired output size after convolution
        # # padding = ((desired_output_shape[0] * stride[0] - input_shape[0] + 1) // 2, (desired_output_shape[1] * stride[1] - input_shape[1] + 1) // 2)
        # print('padding', padding)

        # # Define the convolutional layer with the calculated parameters
        # downsampling_conv = nn.Conv2d(in_channels=input_channels, out_channels=3, kernel_size=3, stride=stride, padding=padding)
        # downsampling_conv.bias.data = downsampling_conv.bias.data.to(torch.float16)
        # downsampling_conv.weight.data = downsampling_conv.weight.data.to(torch.float16)
        # downsampled_depth_map = downsampling_conv(depth_images.to(torch.float16))

        # print("Original depth map shape:", depth_images.shape)
        # print("Downsampled depth map shape:", downsampled_depth_map.shape)


        # # Depth Mask
        # # Step 1: Trilinear operation to compute self-attention matrix
        # # Reshape f_depth to (batch_size, num_channels, height * width)
        # # print(depth_images.shape) # [32, 3, 336, 336]
        # f_depth=depth_images
        # print('f_depth', f_depth.shape) # type tensor
        # batch_size= depth_images.shape[0]
        # num_channels= depth_images.shape[1]
        # f_depth_flat = f_depth.view(batch_size, num_channels, -1)
        # print('f_depth_flat', f_depth_flat.shape)
        # print('f_depth_flat T', f_depth_flat.transpose(1, 2).shape)
        # # Compute the outer product of f_depth_flat with its transpose
        # attention_matrix = torch.bmm(f_depth_flat, f_depth_flat.transpose(1, 2))
        # print('attention_matrix ',attention_matrix)
        # attention_matrix = attention_matrix.to(torch.float32)
        # print('attention_matrix ',attention_matrix)

        # # Step 2: Apply softmax twice
        # attention_matrix_softmax1 = nn.functional.softmax(attention_matrix, dim=1)
        # attention_matrix_softmax2 = nn.functional.softmax(attention_matrix_softmax1, dim=2)

        # # Convert the tensor to a floating-point data type
        # attention_matrix_softmax2_float = attention_matrix_softmax2.to(torch.float32)
        # print('attention_matrix_softmax2_float ', attention_matrix_softmax2_float.shape)

        # # Step 3: Element-wise multiply to obtain f_mask
        # f_mask = torch.mul(f_depth, attention_matrix_softmax2_float.unsqueeze(1))
        # print(f_mask)
        # # Normalize f_mask
        # f_mask = nn.functional.normalize(f_mask, p=2, dim=1)
        # print(f_mask)

        # # Method Learnable Fusion Layer (e.g., fully connected layer)
        # # input_dtype = image_features.dtype # float32
        # input_dtype = torch.bfloat16# bfloat16
        # print(input_dtype)
        # # concat_features= torch.cat((image_features.to(input_dtype), image_features_depth.to(input_dtype)), dim=1) #[16, 1152, 1024]
        # concat_features= torch.cat((image_features.to(input_dtype), resized_depth_map.to(input_dtype)), dim=2) #[16, 576, 2048]
        # print(concat_features.shape)
        # # fused_embedding_size= concat_features.size(-1)#2048  # 13b -> 
        # # embedding_size= image_features.size(-1)#1024 # 13b -> 
        # image_features = self.fusion_layer(concat_features) # self is DepthLlavaLlamaForCausalLM and  self.get_model() is LLavaLLamaModel
        # print('image_features', image_features.shape)



        # Method extend embedding with extra info



        image_features = self.get_model().mm_projector(image_features) #[32, 576, 1024]
        print("image_features na projector", image_features.shape)  #[32, 576, 4096]
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, depth_images, image_sizes=None
    ):
        print("prepare_inputs_labels_for_multimodal in LlavaMetaForCausalLM in arch method2")
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                print(" hierr extra check in arch method 2")
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                print('need extra check from arch in arch method2')
            print('--- Let op images are a list')
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(concat_images.device) for x in image_features]

            # concat_images_depth = torch.cat([image for image in depth_images], dim=0)
            # image_features_depth = self.encode_images(concat_images_depth)
            # split_sizes = [image.shape[0] for image in depth_images]
            # image_features_depth = torch.split(image_features_depth, split_sizes, dim=0)
            # image_features_depth = [x.flatten(0, 1).to(concat_images_depth.device) for x in image_features_depth]  
        else:
            image_features = self.encode_images(images,depth_images)
            # image_features_depth = self.encode_images(depth_images)


        # print("image_features in arch ", image_features.shape) #torch.Size([1, 576, 4096])
        # print("image_features_depth in arch ", image_features_depth.shape) #torch.Size([1, 576, 4096])
        
        # print("image_features in arch before fusion layer ", image_features) #torch.Size([1, 576, 4096])

        # I added RGB and Depth fusion
        # print('image_features shape ',image_features.shape) #torch.Size([32, 576, 4096])
        # print('image_features_depth shape ',image_features_depth.shape) #torch.Size([32, 576, 4096])
        # image_features_depth = image_features_depth.mean(dim=1, keepdim=True) # Make grayscale

        # # Method Learnable Fusion Layer (e.g., fully connected layer)
        # input_dtype = image_features.dtype # bfloat16
        # concat_features= torch.cat((image_features.to(input_dtype), image_features_depth.to(input_dtype)), dim=1) #[16, 1152, 4096]
        # concat_features= torch.cat((image_features.to(input_dtype), image_features_depth.to(input_dtype)), dim=2) #[16, 576, 8192]
        # # fused_embedding_size= concat_features.size(-1)#8192  # 13b -> 10240
        # # embedding_size= image_features.size(-1)#4096 # 13b -> 5120
        # image_features = self.fusion_layer(concat_features) # self is DepthLlavaLlamaForCausalLM and  self.get_model() is LLavaLLamaModel

        # Method Concatenation with Learnable Weights
        # fusion_layer_weights = torch.nn.Parameter(torch.randn(embedding_size*2, fused_embedding_size))
        # print("self.fusion_layer_weights in arch", self.fusion_layer_weights)
        # print(torch.cat((image_features, image_features_depth), dim=1).shape) #[32, 1152, 4096]
        # image_features = torch.matmul(torch.cat((image_features, image_features_depth), dim=1), self.fusion_layer_weights)

        # # Method Attention Mechanism
        # # fusion_layer = torch.nn.Linear(embedding_size*2, 1)
        # query = self.fusion_layer(torch.cat((image_features, image_features_depth), dim=1))
        # # query = self.fusion_layer(torch.cat((image_features, image_features_depth), dim=2))
        # attention_weights = nn.functional.softmax(query, dim=1)
        # print("attention_weights", attention_weights.shape)
        # image_features = torch.sum(torch.cat((image_features, image_features_depth), dim=1) * attention_weights, dim=1)
        # # image_features = torch.sum(torch.cat((image_features, image_features_depth), dim=2) * attention_weights, dim=1)

        # # print("image_features in arch ", image_features)
        # print("image_features in arch ", image_features.shape) #torch.Size([1, 576, 4096]) or train torch.Size([32, 576, 4096]) 
        



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

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # The projection layer is implicitly applied 
        # when obtaining the text space representation by passing the input through the 
        # model's embedding layer and concatenating the resulting embeddings with the image features.
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                # i added because of not same device error
                cur_image_features= cur_image_features.to(self.device)
                cur_input_embeds_1= cur_input_embeds_1.to(self.device)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
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
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            # i added because of not same device error
            # print(cur_labels.device)
            cur_new_input_embeds = [embed.to(cur_labels.device) for embed in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
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

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        print('initialize_vision_tokenizer')
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            print("hier1")

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg
                print("hier2")

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
                print("hier3")

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
                print("hier4")
                
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            print("hier5")

from datasets import load_dataset
# from scr.eval import evaluate
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, BatchSampler, SequentialSampler
import requests
import PIL
# from PIL import Image
from io import BytesIO

import os
import random
# import torch.optim as optim
import json
import numpy as np
from sklearn.model_selection import train_test_split

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import re
from collections import defaultdict

depth_feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")



def make_captions(true_caption, relation, relevant_relations):
    usage_counter = defaultdict(int)
    # relevant_relations  = ['in front of', 'behind', 'left of', 'left side of', 'right of', 'right side of', 'under', 'on', 'below', 'above', 'in', 'beneath', 'next to', 'far away from', 'by', 'beside', 'at the side', 'between']
    relevant_relations  = ['in front of', 'behind', 'left of', 'right of', 'under', 'on', 'below', 'above', 'in', 'beneath', 'next to', 'far away from', 'by', 'beside', 'at the side', 'between']
    same_concepts = [['left of', 'left side of'], ['right of', 'right side of'], ['under', 'below', 'beneath'], ['on', 'above'], ['next to', 'by', 'beside', 'at the side']]

    # Find the concept group of the given relation
    concept_group = None
    for group in same_concepts:
        if relation in group:
            concept_group = group
            break

    # If the relation is in a concept group, exclude the whole group from other_relations
    if concept_group:
        other_relations = [rel for rel in relevant_relations if rel not in concept_group]
    else:
        other_relations = [rel for rel in relevant_relations if rel != relation]

    # Sort other_relations by their usage count (least used first)
    other_relations.sort(key=lambda rel: usage_counter[rel])

    # Initialize a list to store the selected relations
    selected_relations = []
    selected_concept_groups = set()

    if relation not in usage_counter:
        usage_counter[relation] = 0

    # Select relations ensuring no two from the same concept group
    for rel in other_relations:
        rel_concept_group = None
        for group in same_concepts:
            if rel in group:
                rel_concept_group = tuple(group)
                break
        
        if rel_concept_group not in selected_concept_groups:
            selected_relations.append(rel)
            if rel_concept_group:
                selected_concept_groups.add(rel_concept_group)
        
        if len(selected_relations) == 3:
            break
    
    captions = [true_caption]
    # selected_relations = random.sample(other_relations[:len(relevant_relations)-len(same_concepts)], 3)
    # while any([any([rel in group for group in same_concepts]) for rel in selected_relations]):
    #     selected_relations = random.sample(other_relations[:len(relevant_relations)-len(same_concepts)], 3)

    for new_relation in selected_relations:
        new_caption = re.sub(r'\b' + re.escape(relation) + r'\b', new_relation, true_caption)
        captions.append(new_caption)
        # Update the usage counter
        usage_counter[new_relation] += 1

    random.shuffle(captions)
    label = captions.index(true_caption)

    # Map to ABCD
    label_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    mapped_label = label_mapping[label]

    # Update the usage counter for the true relation
    usage_counter[relation] += 1

    return captions, mapped_label






# def make_captions(true_caption, relation, relevant_relations):
#     relevant_relations  = ['in front of', 'behind', 'left of', 'left side of', 'right of', 'right side of', 'under', 'on','below', 'above','in', 'beneath', 'next to', 'far away from', 'by', 'beside', 'at the side' ,'between' ]
#     same_concepts=[['left of', 'left side of'],['right of', 'right side of'],['under','below','beneath'],['on', 'above' ], ['next to','by', 'beside', 'at the side']]
    
#     # other_relations = list(relevant_relations)
#     # other_relations.remove(relation) if relation in other_relations else None
#     # other_relations = [rel for rel in other_relations if relation not in rel]

#      # Find the concept group of the given relation
#     concept_group = None
#     for group in same_concepts:
#         if relation in group:
#             concept_group = group
#             break

#     # If the relation is in a concept group, exclude the whole group from other_relations
#     if concept_group:
#         other_relations = [rel for rel in relevant_relations if rel not in concept_group]
#     else:
#         other_relations = [rel for rel in relevant_relations if rel != relation]
#     # print(relation, other_relations)
    
    
#     captions=[true_caption]
#     selected_relations = random.sample(other_relations, 3)
#     while any([any([rel in group for group in same_concepts]) for rel in selected_relations]):
#         selected_relations = random.sample(other_relations, 3)

#     for new_relation in selected_relations:
#         # captions.append(true_caption.replace(relation, new_relation))
#         new_caption = re.sub(r'\b' + re.escape(relation) + r'\b', new_relation, true_caption)
#         captions.append(new_caption)
        
#     random.shuffle(captions)
#     # print('captions: ', captions)
#     # print(true_caption)
#     label= captions.index(true_caption)

#     # map to ABCD
#     label_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
#     mapped_label = label_mapping[label]

#     return captions, mapped_label

def make_captions_front_behind(true_caption, relation, relevant_relations):
    captions=[true_caption]
    other_relations = list(relevant_relations)
    other_relations.remove(relation) if relation in other_relations else None
    other_relations = random.sample(other_relations, 2)
    for new_relation in other_relations:
        captions.append(true_caption.replace(relation, new_relation))
    random.shuffle(captions)
    # print(captions)
    # captions = random.sample(captions, len(captions))
    label= captions.index(true_caption)

    # map to ABCD
    label_mapping = {0: 'A', 1: 'B', 2: 'C'}
    mapped_label = label_mapping[label]

    return captions, mapped_label
        
def download_image_to_np(url):
    if url.startswith("http") or url.startswith("https"):
        response = requests.get(url, stream=True)
        image = PIL.Image.open(response.raw)
    else:
        image = PIL.Image.open(url)
    image_np = np.array(image)
    # Ensure the image is in RGB format
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    return image_np

def download_image(url):
    if url.startswith("http") or url.startswith("https"):
        response = requests.get(url, stream=True)
        image = PIL.Image.open(response.raw)
    else:
        image = PIL.Image.open(url)
    return image

def get_depth_map(image):
    image= download_image_to_np(image)
    pixel_values = depth_feature_extractor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = depth_model(pixel_values)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=(image.shape[0], image.shape[1]),  # Set size to match image dimensions
                        mode="bicubic",
                        align_corners=False,
                ).squeeze()
    
    output = prediction.cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype('uint8')
    formatted_tensor = torch.from_numpy(formatted)
    formatted_tensor = formatted_tensor.unsqueeze(dim=-1)
    depth = PIL.Image.fromarray(formatted)
    # depth.show()
    return depth, formatted_tensor
    

def extract_relation(true_captions):
    check_relations = ['in front of', 'behind', 'left of', 'right of', 'under', 'on', 'above']
    relations=[]
    for true_caption in true_captions:
        for relation in check_relations:
            if relation in true_caption:
                relations.append(relation)
                break
    if len(relations)!=len(true_captions):
        print('--- Not all relations of captions found')
        print(true_captions)
    return relations  

def extract_relation2(true_captions):
    check_relations = ['in front of', 'behind', 'left of', 'right of', 'under', 'on','below', 'above','in']
    relations=[]
    relation_not_found=[]
    for true_caption in true_captions:
        for relation in check_relations:
            rel_found=False
            if relation in true_caption:
                rel_found=True
                relations.append(relation)
                break
        if rel_found==False:
            relation_not_found.append(true_caption)

    if len(relations)!=len(true_captions):
        print('--- Not all relations of captions found')
        print(relation_not_found[:10])
        print('------------',len(relation_not_found))
    return relations  


def extract_relation3(true_caption):
    check_relations = ['in front of', 'behind', 'left of', 'left side of', 'right of', 'right side of', 'under', 'on','below', 'above','in', 'beneath ', 'next to', 'far away from', 'by', 'beside', 'at the side' ,'between' ]
    for relation in check_relations:
        rel_found=False
        if relation in true_caption:
            rel_found=True
            break
    if rel_found==False:
        # print('rel not found: ', true_caption)
        relation = 'other relation'
    return relation 

################################ Dataloaders ##################################################

class VSRDataset_front_behind(Dataset):
    def __init__(self, hf_dataset, split):
        self.hf_dataset = hf_dataset
        self.split= split
        self.root_folder = '/project/msc-thesis-project/all_vsr_depth'
        self.image_folder = '/project/msc-thesis-project/all_vsr_images'
        self.relevant_relations =['in front of', 'behind of', 'at a similar level as'] # =['above', 'below', 'left of', 'right of']

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Define how to retrieve a sample
        relation= self.hf_dataset[idx]['relation']
        label= self.hf_dataset[idx]['label']
        if relation in self.relevant_relations and label==1:
            image_link= self.hf_dataset[idx]['image_link']
            image_name = image_link.split('/')[-1]
            depth_map= os.path.join(self.root_folder, image_name)
            if os.path.exists(depth_map):
                caption = self.hf_dataset[idx]['caption']
                captions, position_label = make_captions_front_behind(caption, relation,self.relevant_relations)
                if len(captions)==3:
                    image_link= os.path.join(self.image_folder, image_name)
                    return image_link, depth_map, captions, position_label
                else:
                    print(f"len(captions) is {len(captions)};  {captions}")
                    return None
            else:
                print("--- no depth map for ", image_name)
                return None
        else:
            return None
        
class VSRDataset(Dataset):
    def __init__(self, hf_dataset, split):
        self.hf_dataset = hf_dataset
        self.split= split
        self.root_folder = '/project/msc-thesis-project/all_vsr_depth'
        self.image_folder = '/project/msc-thesis-project/all_vsr_images'
        # self.relevant_relations =['in front of', 'behind of', 'next to', 'above', 'below', 'left of', 'right of']
        self.relevant_relations  = ['in front of', 'behind', 'left of', 'left side of', 'right of', 'right side of', 'under', 'on','below', 'above','inside', 'in', 'beneath ', 'next to', 'far away from', 'by', 'beside', 'at the side' ,'between' ]
    

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Define how to retrieve a sample
        relation= self.hf_dataset[idx]['relation']
        label= self.hf_dataset[idx]['label']
        if relation in self.relevant_relations and label==1:
            image_link= self.hf_dataset[idx]['image_link']
            image_name = image_link.split('/')[-1]
            depth_map= os.path.join(self.root_folder, image_name)
            if os.path.exists(depth_map):
                caption = self.hf_dataset[idx]['caption']
                captions, position_label = make_captions(caption, relation,self.relevant_relations)
                if len(captions)==4:
                    image_link= os.path.join(self.image_folder, image_name)
                    return image_link, depth_map, captions, position_label
                else:
                    print(f"len(captions) is {len(captions)};  {captions}")
                    return None
            else:
                print("--- no depth map for ", image_name)
                return None
        if label==0: # remove before dataloader
            print('label==0')
            # return None

class VSR_TF_f_Dataset(Dataset):
    def __init__(self, hf_dataset, split):
        self.hf_dataset = hf_dataset
        self.split= split
        self.root_folder = '/project/msc-thesis-project/all_vsr_depth'
        self.image_folder = '/project/msc-thesis-project/all_vsr_images'
        self.relevant_relations  = ['in front of', 'behind', 'left of', 'left side of', 'right of', 'right side of', 'under', 'on','below', 'above','in', 'beneath ', 'next to', 'far away from', 'by', 'beside', 'at the side' ,'between' ]

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Define how to retrieve a sample
        image_link= self.hf_dataset[idx]['image_link']
        image_name = image_link.split('/')[-1]
        depth_map= os.path.join(self.root_folder, image_name)
        relation= self.hf_dataset[idx]['relation']

        if relation in self.relevant_relations:
            if os.path.exists(depth_map):
                label= self.hf_dataset[idx]['label']
                caption = self.hf_dataset[idx]['caption']
                image_link= os.path.join(self.image_folder, image_name)
                return image_link, depth_map, caption, label
            else:
                print("--- no depth map for ", image_name)
                # depth, formatted_tensor = get_depth_map(image_link)
                # depth.save(depth_map)
                return None
        else:
            return None
        

class VSR_TF_Dataset(Dataset):
    def __init__(self, hf_dataset, split):
        self.hf_dataset = hf_dataset
        self.split= split
        self.root_folder = '/project/msc-thesis-project/all_vsr_depth'
        self.image_folder = '/project/msc-thesis-project/all_vsr_images'

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Define how to retrieve a sample
        image_link= self.hf_dataset[idx]['image_link']
        image_name = image_link.split('/')[-1]
        depth_map= os.path.join(self.root_folder, image_name)

        if os.path.exists(depth_map):
            label= self.hf_dataset[idx]['label']
            caption = self.hf_dataset[idx]['caption']
            image_link= os.path.join(self.image_folder, image_name)
            return image_link, depth_map, caption, label
        else:
            print("--- no depth map for ", image_name)
            # depth, formatted_tensor = get_depth_map(image_link)
            # depth.save(depth_map)
            return None
        

class COCO_Dataset(Dataset):
    def __init__(self, hf_dataset, split):
        self.hf_dataset = hf_dataset
        self.split= split
        self.root_folder = '../../vsr_depth'
        self.relevant_relations =[' at similar level as ',' behind of ', ' in front of ']

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Define how to retrieve a sample
        relation= self.hf_dataset[idx]['relation']
        label= self.hf_dataset[idx]['label']
        if relation in self.relevant_relations and label==1:
            image_link= self.hf_dataset[idx]['image_link']
            image_name = image_link.split('/')[-1]
            depth_map= os.path.join(self.root_folder, self.split, image_name)
            if os.path.exists(depth_map):
                caption = self.hf_dataset[idx]['caption']
                captions, position_label = make_captions(caption, relation,self.relevant_relations)
                if len(captions)==4:
                    return image_link, depth_map, captions, position_label
                else:
                    print(f"len(captions) is {len(captions)};  {captions}")
                    return None
            else:
                print("--- no depth map for ", image_name)
                return None
        if label==0: # remove before dataloader
            print('label==0')
            # return None


class WhatsupControlDataset(Dataset):
    def __init__(self, hf_dataset, split):
        self.hf_dataset = hf_dataset
        self.split= split
        # print('split', split)
        self.root_folder = '/project/msc-thesis-project/forked_repos/whatsup_vlms/data/controlled_depth_images' 
        # self.relevant_relations =['above', 'below', 'left of', 'right of']
        # self.relevant_relations =['in front of', 'behind of', 'at a similar level as']

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Define how to retrieve a sample
        data_folder=self.hf_dataset[idx]['image_path']
        image_path= f'/project/msc-thesis-project/forked_repos/whatsup_vlms/{data_folder}'
        image_name= os.path.basename(image_path)
        depth_map= os.path.join(self.root_folder, image_name)
        caption_options= self.hf_dataset[idx]['caption_options']
        true_caption= caption_options[0]

        captions = random.sample(caption_options, len(caption_options))
        label= captions.index(true_caption)

        # map to ABCD
        label_mapping = {0: 'A',1: 'B',2: 'C',3: 'D'}
        position_label = label_mapping[label]

        return image_path, depth_map, captions, position_label
    

class WhatsupDataset(Dataset):
    def __init__(self, hf_dataset, split):
        self.hf_dataset = hf_dataset
        self.split= split
        # print('split', split)
        self.root_folder =  '/project/msc-thesis-project/forked_repos/whatsup_vlms/data/controlled_depth_images'
        # self.relevant_relations =['above', 'below', 'left of', 'right of']
        # self.relevant_relations =['in front of', 'behind of', 'at a similar level as']

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Define how to retrieve a sample
        image_name=self.hf_dataset[idx][0]
        # padd with zeros in front to make it 12 digits long
        image_name = str(image_name).zfill(12)
        image_path= f'/project/msc-thesis-project/forked_repos/whatsup_vlms/data/val2017/{image_name}.jpg'
        depth_map= os.path.join(self.root_folder, f'{image_name}.jpg')
        caption_options=[self.hf_dataset[idx][1], self.hf_dataset[idx][2]]
        true_caption= self.hf_dataset[idx][1]

        captions = random.sample(caption_options, len(caption_options))
        label= captions.index(true_caption)

        # map to ABCD
        label_mapping = {0: 'A',1: 'B'}
        position_label = label_mapping[label]

        return image_path, depth_map, captions, position_label
    

class VSD_Dataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
        self.root_folder = '../../vsd_depth'

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Define how to retrieve a sample
        sample= self.hf_dataset[idx]
        # subject = sample['s']
        # object = sample['o']
        # groundtruth_relation = sample['r']
        captions =  sample['captions']
        position_label= sample['position_label']
        # img_id =  sample['img_id']
        # anno_img_path = f'/project/Anno_VSD_images/{img_id}.jpg'
        image_link= sample['image_link']
        depth_map = sample['depth_map']
        return image_link, depth_map, captions, position_label
    

class clevrDataset(Dataset):
    def __init__(self, hf_dataset, split):
        self.hf_dataset = hf_dataset
        self.split= split
        self.root_folder = f'/project/clevr_depth/{self.split}' 

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Define how to retrieve a sample
        image_id=self.hf_dataset[idx]['image_id']
        image_path= f'/project/clevr/{self.split}/image{image_id}.jpg'
        image_name= f'image{image_id}.jpg'
        depth_map= os.path.join(self.root_folder, image_name)
        captions= self.hf_dataset[idx]['captions']
        label= self.hf_dataset[idx]['label']
        return image_path, depth_map, captions, label

                        
def custom_collate_fn(batch):
    # print(batch)
    if batch is not None:
        batch = [sample for sample in batch if sample is not None]
        # print('batch', batch)
        batch_imgs, batch_depths, batch_caps, labels = zip(*batch)
        return batch_imgs, batch_depths, batch_caps, labels



################################ get functions ##################################################

def get_COCO_dataloader(task, split):
    with open(f"/project/annotations/instances_{split}2017.json") as f:
        dataset = json.load(f)

    dataset_name='COCO'
    # Create DataLoader instances
    batch_size = 32
    if task=='front_behind':
        data_loader = DataLoader(COCO_Dataset(dataset, f'{split}'), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    output_file_path = f"./playground/data/eval/custom2/{dataset_name}_{split}_{task}.json"
    return data_loader, output_file_path



def get_VSR_dataloader(task, split, data_split="zeroshot"):
    
    data_files = {"train": "train.jsonl", "val": "dev.jsonl", "test": "test.jsonl"}
    if data_split == "zeroshot":
        # dataset = load_dataset("cambridgeltl/vsr_zeroshot", data_files=data_files) # more difficult than vsr_random
        dataset = load_dataset("cambridgeltl/vsr_zeroshot") # more difficult than vsr_random
        dataset_name='VSR'
    else:
        # dataset = load_dataset("cambridgeltl/vsr_random", data_files=data_files) # less difficult
        dataset = load_dataset("cambridgeltl/vsr_random") 
        dataset_name='VSR_random'

    batch_size = 32
    if task=='TF':
        data_loader = DataLoader(VSR_TF_Dataset(dataset[f'{split}'], f'{split}'), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    elif task=='TF_f':
        data_loader = DataLoader(VSR_TF_f_Dataset(dataset[f'{split}'], f'{split}'), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    elif task== 'classification':
        data= dataset[f'{split}']
        new_data = [d for d in data if d['label']==1]
        for d in new_data:
            if d['label']==0:
                print('d[label]==0', d)
        data_loader = DataLoader(VSRDataset(new_data, f'{split}'), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    else:
        task='front_behind_classification'
        data_loader = DataLoader(VSRDataset_front_behind(dataset[f'{split}'], f'{split}'), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    output_file_path = f"./playground/data/eval/custom2/{dataset_name}_{split}_{task}4.json"
    return data_loader, output_file_path


def get_whatsup_dataloader(split, dataset_type= "controlled_images"):

    if dataset_type== "controlled_images" or dataset_type== "controlled_images_new":
        with open('/project/msc-thesis-project/forked_repos/whatsup_vlms/data/controlled_images_dataset.json') as f:
            subdataset_A = json.load(f)
        with open('/project/msc-thesis-project/forked_repos/whatsup_vlms/data/controlled_clevr_dataset.json') as f:
            subdataset_B = json.load(f)
        controlled_dataset = subdataset_A + subdataset_B
        random.shuffle(controlled_dataset)

        total_samples = len(controlled_dataset)
        # train_size = int(0.8 * total_samples)
        # train_dataset, test_dataset = controlled_dataset[:train_size], controlled_dataset[train_size:]
        # print(len(train_dataset), len(test_dataset)) #656 164

        # train_val_size = int(0.8 * total_samples)
        test_size = val_size = int(0.1 * total_samples)

        train_val_dataset, test_dataset = train_test_split(controlled_dataset, test_size=test_size, random_state=42)
        train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=val_size, random_state=42)
        dataset_mapping = {'train': train_dataset, 'test': test_dataset, 'val': val_dataset, 'whole_dataset': controlled_dataset}
    elif dataset_type=="COCO":
        # only val_2017 ?
        with open('/project/msc-thesis-project/forked_repos/whatsup_vlms/data/coco_qa_two_obj.json') as f:
            dataset = json.load(f)
        random.shuffle(dataset)
        dataset_mapping = {'train': train_dataset, 'test': test_dataset, 'val': val_dataset}

    selected_dataset = dataset_mapping.get(split)
    dataset_name='whatsup'
    task= 'classification'
    
    # true_captions = [item['caption_options'][0] for item in selected_dataset]
    # relations = extract_relation(true_captions)
    # class_labels, class_sample_count = np.unique(relations, return_counts=True)
    # print(class_labels, class_sample_count )

    shuffle = False if split=='test' or split=='val' else True
    # shuffle = False 
    batch_size = 32
    if dataset_type== "controlled_images"or dataset_type== "controlled_images_new":
        data_loader = DataLoader(WhatsupControlDataset(selected_dataset, f'{split}'), batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)
    else:
        data_loader = DataLoader(WhatsupDataset(selected_dataset, f'{split}'), batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)
    output_file_path = f"./playground/data/eval/custom2/{dataset_name}_{split}_{task}_{dataset_type}.json" #_balanced
    return data_loader, output_file_path


# def get_clevr_dataloader(split):
#     # with open(f"clevr_whole_dataset_{split}.json", 'r') as json_file:
#     #       dataset= json.load(json_file)

#     images=dataset_dict["image"]
#     depth_maps=dataset_dict["depth"]
#     prompts=dataset_dict["prompt"]

#     return images,depth_maps, prompts

def get_clevr_dataloader(v_dataset,split):
    with open(f"/project/{v_dataset}_{split}.json", 'r') as json_file:
          dataset= json.load(json_file)

    # Create DataLoader instances
    batch_size = 32
    data_loader = DataLoader(clevrDataset(dataset, split=split), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    output_file_path = f"./playground/data/eval/custom2/{v_dataset}_{split}.json"
    return data_loader, output_file_path

def get_VSD_dataloader(split):
    # this is prepossesed
    # with open(f"/project/VSDv2/newvsd_{split}.json", 'r') as f:
    #     dataset = json.load(f)
    
    with open(f"/project/VSD_for_llava/vsd_{split}.json", 'r') as f:
        dataset = json.load(f)
    dataset_name='VSD'
    task='classification'
    # Create DataLoader instances
    batch_size = 32
    data_loader = DataLoader(VSD_Dataset(dataset), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    output_file_path = f"./playground/data/eval/custom2/{dataset_name}_{split}_{task}.json"
    return data_loader, output_file_path




def make_llava_format(dataset_name, data_loader, output_file_path, prompt_template=None):
    """
    ormats data from the given data loader into the LLaVA format and writes it to the specified output file.

    Args:
        data_loader (DataLoader): The data loader containing the input data to be formatted.
        output_file_path (str): The file path where the formatted data will be written.
        prompt (str, optional): An optional prompt to be included in the formatted data. Defaults to None.

    Returns:
        None

    Raises:
        ValueError: If the output_file_path is invalid or if the data loader is empty.

    Notes:
        - The data loader should contain the necessary information to be formatted in the LLaVA format.
        - The output file will be written in the specified file path.
        - If a prompt is provided, it will be included in the formatted data.

    Example:
        >>> make_llava_format(data_loader, "output.txt", prompt="Please answer the following questions:")
   
    """
    
    # label_mapping = {'A': 0, 'B':1, 'C':2,'D':3}
    TF_mapping= {1: 'True', 0: 'False'}

    data=[]
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, sample in pbar:
        batch_imgs, batch_depths, batch_caps, position_labels = sample

        # if dataset_name=='VSR_TF':
        #     # print(position_labels)
        #     class_labels, class_sample_count = np.unique(position_labels, return_counts=True)
        #     print(class_labels, class_sample_count ) #[0 1] [12 20] not balanced
        # elif dataset_name=='whatsup_COCO':
        #     true_captions= [cap[label_mapping[position_labels[k]]]for k, cap in enumerate(batch_caps)]
        #     relations = extract_relation2(true_captions)
        #     class_labels, class_sample_count = np.unique(relations, return_counts=True)
        #     print(class_labels, class_sample_count ) # e.g ['above' 'below' 'left of' 'on' 'right of'] [ 3  5 12  6  6] and ['above' 'below' 'left of' 'right of'] [ 8  4 11  9], so no 'on' relation in this batch
        # else:
        #     true_captions= [cap[label_mapping[position_labels[k]]]for k, cap in enumerate(batch_caps)]
        #     relations = extract_relation(true_captions)
        #     class_labels, class_sample_count = np.unique(relations, return_counts=True)
        #     print(class_labels, class_sample_count )


        for n, batch_cap in enumerate(batch_caps):
            img_path=batch_imgs[n]
            img_file_name = os.path.basename(img_path)
            depth_path = batch_depths[n]

            if dataset_name=='VSR_TF_f':
                position_label= TF_mapping[position_labels[n]]
            elif dataset_name!='VSR_TF':
                position_label= position_labels[n]
            else:
                position_label= TF_mapping[position_labels[n]]
            # if len(position_label) <2:

            prompt= prompt_template(batch_cap)
            # Add another sample
            data.append({
                "id": img_file_name,
                "image": img_path,
                "depth": depth_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": prompt
                    },
                    {
                        "from": "gpt",
                        "value": position_label
                    }
                ]
            })
                
    with open(output_file_path, "w") as json_file:
        json.dump(data, json_file, indent=2)
    print(f"JSON data written to {output_file_path}")


################################ prompts ##################################################

def VSR_TF_prompt(batch_cap):
    return f"<image>\nGiven the image, is the following statement True or False?\n{batch_cap}\nAnswer the question using a single word."

def pick_from_two_captions_prompt(batch_cap):
    # return f"Which of these 2 relations best describes the relation in the given image:\nA: {batch_cap[0]}\nB: {batch_cap[1]}\n"
    # return f"Identify the relation that best describes the image:\nA: {batch_cap}\nB: {batch_cap}\nChoose the most suitable option."
    # return f"Identify the correct relation between the objects mentioned, given the image:\nA: {batch_cap[0]}\nB: {batch_cap[1]}\nChoose the most suitable option.\n"
    # return f"Identify the correct relation between the objects mentioned, given the image:\nA: {batch_cap[0]}\nB: {batch_cap[1]}\nChoose the most suitable option.\n"
    return f"<image>\nWhat is the spatial relationship between the objects in the image?\nA: {batch_cap[0]}\nB: {batch_cap[1]}\nAnswer with the option's letter from the given choices directly."


def pick_from_four_captions_prompt(batch_cap):
    # return f"Which of these 4 relations best describes the given image:\nA: {batch_cap[0]}\nB: {batch_cap[1]}\nC: {batch_cap[2]}\nD: {batch_cap[3]}\n"
    # return f"Identify the correct relation between the objects mentioned, given the image:\nA: {batch_cap[0]}\nB: {batch_cap[1]}\nC: {batch_cap[2]}\nD: {batch_cap[3]}\nChoose the most suitable option and please respond with only the letter corresponding to the correct answer, e.g., 'A', 'B', 'C', or 'D'.\n"
    return f"<image>\nWhat is the spatial relationship between the objects in the image?\nA: {batch_cap[0]}\nB: {batch_cap[1]}\nC: {batch_cap[2]}\nD: {batch_cap[3]}\nAnswer with the option's letter from the given choices directly."

def pick_from_three_captions_prompt(batch_cap):
    # return f"Identify the correct relation between the objects mentioned, given the image:\nA: {batch_cap[0]}\nB: {batch_cap[1]}\nC: {batch_cap[2]}\nChoose the most suitable option.\n"
    # return f"What is the correct spatial relationship between the mentioned objects, as shown in the image? \nA: {batch_cap[0]}\nB: {batch_cap[1]}\nC: {batch_cap[2]}\nAnswer with the option's letter from the given choices directly.\n"
    return f"<image>\nWhat is the spatial relationship between the objects in the image?\nA: {batch_cap[0]}\nB: {batch_cap[1]}\nC: {batch_cap[2]}\nAnswer with the option's letter from the given choices directly."




if __name__=="__main__":

    random.seed(40)

    # data_loader, output_file_path = get_clevr_dataloader("clevr",'val')
    # make_llava_format("clevr", data_loader, output_file_path, prompt_template=pick_from_three_captions_prompt)

    # data_files = {"train": "train.jsonl", "val": "dev.jsonl", "test": "test.jsonl"}
    # dataset = load_dataset("cambridgeltl/vsr_zeroshot", data_files=data_files) 

    # data_files = {"train": "train.jsonl", "val": "dev.jsonl", "test": "test.jsonl"}
    # dataset = load_dataset("cambridgeltl/vsr_random", data_files=data_files) 
    # dataset = load_dataset("cambridgeltl/vsr_random")['validation'] 
    

    # data_loader, output_file_path = get_VSR_dataloader("TF", split='test', data_split='random')
    # make_llava_format("VSR_TF", data_loader, output_file_path, prompt_template=VSR_TF_prompt)

    # data_loader, output_file_path = get_VSR_dataloader("TF", split='val')
    # make_llava_format("VSR_TF", data_loader, output_file_path, prompt_template=VSR_TF_prompt)

    # data_loader, output_file_path = get_VSR_dataloader("TF_f", split='train')
    # make_llava_format("VSR_TF_f", data_loader, output_file_path, prompt_template=VSR_TF_prompt)

    # data_loader, output_file_path = get_COCO_dataloader("front_behind_classification", split='test')
    # make_llava_format("COCO_front_behind", data_loader, output_file_path, prompt_template=pick_from_three_captions_prompt)

    data_loader, output_file_path = get_VSR_dataloader("classification", split='train', data_split='random')
    make_llava_format("VSR_random", data_loader, output_file_path, prompt_template=pick_from_four_captions_prompt)

    # data_loader, output_file_path = get_VSR_dataloader("classification", split='validation')
    # make_llava_format("VSR", data_loader, output_file_path, prompt_template=pick_from_four_captions_prompt)
    
    
    # data_loader, output_file_path = get_whatsup_dataloader(split='test', dataset_type= "controlled_images_new")
    # make_llava_format("whatsup_controlled", data_loader, output_file_path, prompt_template=pick_from_four_captions_prompt)

    # data_loader, output_file_path = get_whatsup_dataloader(split='val', dataset_type= "COCO")
    # make_llava_format("whatsup_COCO", data_loader, output_file_path, prompt_template=pick_from_two_captions_prompt)

    # data_loader, output_file_path = get_VSD_dataloader('test')
    # make_llava_format("VSD", data_loader, output_file_path, prompt_template=pick_from_four_captions_prompt)




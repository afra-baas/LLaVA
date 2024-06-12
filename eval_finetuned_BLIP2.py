# source: https://huggingface.co/blog/blip-2#using-blip-2-with-hugging-face-transformers
from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoTokenizer, BlipImageProcessor, AutoModelForVisualQuestionAnswering, Blip2Config, AutoModel, AutoConfig
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

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
from llava.conversation import conv_templates


def eval_model(args, model_name, model, processor, context_len):

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        if i<10:
            idx = line["id"]
            question = line['conversations'][0]
            qs = question['value'].replace('<image>', '').strip()
            cur_prompt = qs

            if 'image' in line:
                image_file = line["image"]
                if image_file.startswith("http") or image_file.startswith("https"):
                    response = requests.get(image_file)
                    image = Image.open(BytesIO(response.content))
                else:
                    image = Image.open(os.path.join(args.image_folder, image_file))
                if getattr(model.config, 'mm_use_im_start_end', False):
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                cur_prompt = '<image>' + '\n' + cur_prompt
            else:
                print('It doesnt come here unless', line)

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = processor(image, text=prompt, return_tensors="pt")#.to(device, torch.float16)
            input_ids['max_new_tokens'] = context_len #100

            output_ids = model.generate(**input_ids)
            outputs = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()
        else:
            break
        
    ans_file.close()



def download_image(url):
    if url.startswith("http") or url.startswith("https"):
        response = requests.get(url, stream=True)
        image = Image.open(response.raw)
    else:
        image = Image.open(url)
    return image


def save_with_caption(images, file_name, caption):
    # Create a new blank image for the collage
    collage_width = len(images)* 400
    collage_height = 425
    collage = Image.new('RGB', (collage_width, collage_height))

    for i, image in enumerate(images):
        # Resize the images to a fixed size (you can adjust this as needed)
        image = image.resize((400, 400))
        # Paste the images onto the collage
        collage.paste(image, (i * 400, 0))

    draw = ImageDraw.Draw(collage)
    font = ImageFont.load_default()
    font = font.font_variant(size=11)
    draw.text((5, 405), f'{caption}', fill='black', stroke_width=1,stroke_fill="white", font=font)

    # Save the collage
    collage.save(f'{file_name}')


def eval_single(result_file, data):
    considered_answers=[]
    results = {}
    for line in open(result_file):
        row = json.loads(line)
        question_id = row['question_id']
        if question_id in results:
            results[question_id].append(row)
        else:
            results[question_id] = [row]

    data_type='img'
    type_counts = {}
    correct_counts = {}
    wrong_predictions={'wrong pred': [], 'groundtruth':[], 'row': [], 'question_data':[]}
    for question_data in data:
        type_counts[data_type] = type_counts.get(data_type, 0) + 1
        question_id = question_data['id']
        if question_id not in results:
            correct_counts[data_type] = correct_counts.get(data_type, 0)
            continue
        rows = results[question_id]

        for row in rows:
            # sometime the same image is used for another question, also check something unique, i think answer_id is
            # so pick the next sample with question_id that is not considered yet
            answer_id = row['answer_id']
            if answer_id not in considered_answers:
                considered_answers.append(answer_id)
                pred=row['text'].split(':')[0]
                GT=question_data['conversations'][1]['value']
                if pred == GT:
                    correct_counts[data_type] = correct_counts.get(data_type, 0) + 1

                    # save image in folder 
                    image= download_image(question_data['image'])
                    folderpath= f'{os.path.dirname(result_file)}/correct_images'
                    filename=f'{folderpath}/{question_id}'
                    if not os.path.exists(folderpath):
                        os.makedirs(folderpath)
                    prompt=row['prompt'].split('\n')[2]
                    caption= f'{prompt} Answer: {pred}'
                    save_with_caption([image], filename, caption)
                else:
                    wrong_predictions['wrong pred'].append(row['text'])
                    wrong_predictions['groundtruth'].append(GT)
                    wrong_predictions['row'].append(row)
                    wrong_predictions['question_data'].append(question_data)

                    # save image in folder 
                    image= download_image(question_data['image'])
                    folderpath= f'{os.path.dirname(result_file)}/wrong_images'
                    filename=f'{folderpath}/{question_id}'
                    if not os.path.exists(folderpath):
                        os.makedirs(folderpath)
                    prompt=row['prompt'].split('\n')[2]
                    caption= f'{prompt} Answer: {pred}  GT: {GT}'
                    save_with_caption([image], filename, caption)
                break

    print(correct_counts)
    print(type_counts)
    total_accuracy = correct_counts[data_type] / type_counts[data_type] * 100
    print(f"Total accuracy: {total_accuracy:.2f}%")
    return results, total_accuracy, wrong_predictions


if __name__ == "__main__":

    # Define the directory where your checkpoint is saved
    CKPT= "VSR_TF_epoch3-nodepth-blip2"
    checkpoint_dir = f"./checkpoints/{CKPT}/epoch_2_batch_100"
    model_base ="Salesforce/blip2-opt-2.7b"

    processor = AutoProcessor.from_pretrained(model_base)
    model = Blip2ForConditionalGeneration.from_pretrained(checkpoint_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    test_file="VSR_test_TF.json"
    # CKPT="blip-2_flan"
    root="/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/answers_folder/VSR_TF"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default=f"/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/{test_file}")
    parser.add_argument("--answers-file", type=str, default=f"{root}/{CKPT}/merge.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args(args=[])

    model_name = f'BLIP-2'
    context_len=100
    eval_model(args, model_name, model, processor, context_len)


    # parser2 = argparse.ArgumentParser()
    # parser2.add_argument("--annotation-file", type=str, default=f"/project/msc-thesis-project/forked_repos/LLaVA/playground/data/eval/custom2/{test_file}")
    # parser2.add_argument("--result-file", type=str, default=f"{root}/{CKPT}/merge.jsonl")
    # parser2.add_argument("--result-upload-file", type=str, default=f"{root}/{CKPT}/predictions.jsonl")
    # args2 = parser2.parse_args(args=[])

    # data = json.load(open(args2.annotation_file))
    # results, total_accuracy, wrong_predictions = eval_single(args2.result_file, data)

    # considered_answers=[]
    # with open(args2.result_upload_file, 'w') as fp:
    #     for question in data:
    #         qid = question['id']
    #         rows = results[qid]
    #         for row in rows:
    #             answer_id = row['answer_id']
    #             if answer_id not in considered_answers:
    #                 considered_answers.append(answer_id)
    #                 fp.write(json.dumps({
    #                     'question_id': qid,
    #                     'prediction': row['text']
    #                 }) + '\n')

    #     fp.write(json.dumps({'Total_accuracy': f"{total_accuracy:.2f}%"}) + '\n')
    #     fp.write('\n \n \n')
    #     # TODO: make prettier
    #     fp.write(json.dumps(wrong_predictions) + '\n')


# source: https://huggingface.co/blog/blip-2#using-blip-2-with-hugging-face-transformers
from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoTokenizer, BlipImageProcessor, AutoModelForVisualQuestionAnswering
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



def eval_model(args, model_name, tokenizer, model, image_processor, context_len ):

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            # print('hier')
            image_file = line["image"]
            if image_file.startswith("http") or image_file.startswith("https"):
                response = requests.get(image_file)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
        input_ids['max_new_tokens'] = 100
        # print(input_ids)

        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        # print(input_ids)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        # with torch.inference_mode():
        #     output_ids = model.generate(
        #         input_ids,
        #         images=images,
        #         do_sample=True if args.temperature > 0 else False,
        #         temperature=args.temperature,
        #         max_new_tokens=1024,
        #         use_cache=True,
        #         stopping_criteria=stopping_criteria,
        #     )

        # generated_ids = model.generate(**input_ids, max_new_tokens=10)
        output_ids = model.generate(**input_ids)
        # print(output_ids.shape[1])

        # input_token_len = input_ids['input_ids'].shape[1]
        # input_token_len = input_ids.shape[1]
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
        #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        
        # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        # outputs = outputs.strip()
        outputs = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # prompt for answer
        if args.answer_prompter:
            outputs_reasoning = outputs
            input_ids = tokenizer_image_token(prompt + outputs_reasoning + ' ###\nANSWER:', tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=64,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            outputs = outputs_reasoning + '\n The answer is ' + outputs

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
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
        # sometime the same image is used for another question, also check something unique, i think answer_id is
        # results[row['question_id']] = row
        question_id = row['question_id']
        if question_id in results:
            # Key already exists, append to the existing value
            results[question_id].append(row)
        else:
            # Key doesn't exist, initialize with the new value
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
    # print(considered_answers)
    total_accuracy = correct_counts[data_type] / type_counts[data_type] * 100
    print(f"Total accuracy: {total_accuracy:.2f}%")
    return results, total_accuracy, wrong_predictions


if __name__ == "__main__":

    # source : https://huggingface.co/Salesforce/blip2-flan-t5-xxl
    # processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
    # model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-flan-t5-xxl")

    #https://huggingface.co/Salesforce/blip-vqa-base?library=true

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    image_processor = BlipImageProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # image_processor_class = "BlipImageProcessor"
    # tokenizer_class = "AutoTokenizer"

    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

    from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
    from llava.conversation import conv_templates, SeparatorStyle


    test_file="VSR_test_TF.json"
    CKPT="blip-2"
    root="/project/LLaVA/playground/data/eval/custom2/answers_folder/VSR_TF"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default=f"/project/LLaVA/playground/data/eval/custom2/{test_file}")
    parser.add_argument("--answers-file", type=str, default=f"{root}/{CKPT}/merge.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args(args=[])

    model_name ='BLIP-2'
    context_len=100
    eval_model(args, model_name, tokenizer, model, image_processor, context_len )


    parser2 = argparse.ArgumentParser()
    parser2.add_argument("--annotation-file", type=str, default=f"/project/LLaVA/playground/data/eval/custom2/{test_file}")
    parser2.add_argument("--result-file", type=str, default=f"{root}/{CKPT}/merge.jsonl")
    parser2.add_argument("--result-upload-file", type=str, default=f"{root}/{CKPT}/predictions.jsonl")
    args2 = parser2.parse_args(args=[])

    data = json.load(open(args2.annotation_file))
    results, total_accuracy, wrong_predictions = eval_single(args2.result_file, data)

    considered_answers=[]
    with open(args2.result_upload_file, 'w') as fp:
        for question in data:
            qid = question['id']
            rows = results[qid]
            for row in rows:
                answer_id = row['answer_id']
                if answer_id not in considered_answers:
                    considered_answers.append(answer_id)
                    fp.write(json.dumps({
                        'question_id': qid,
                        'prediction': row['text']
                    }) + '\n')

        fp.write(json.dumps({'Total_accuracy': f"{total_accuracy:.2f}%"}) + '\n')
        fp.write('\n \n \n')
        # TODO: make prettier
        fp.write(json.dumps(wrong_predictions) + '\n')


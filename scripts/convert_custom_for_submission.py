import json
import argparse
from PIL import Image, ImageDraw, ImageFont
import requests
import os
import shutil
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--result-upload-file", type=str)
    parser.add_argument("--TF")
    return parser.parse_args()


def download_image(url):
    if url.startswith("http") or url.startswith("https"):
        response = requests.get(url, stream=True)
        image = Image.open(response.raw)#.convert('RGB')
    else:
        image = Image.open(url)#.convert('RGB')
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


def eval_single(result_file, data, TF):
    
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
    # print('results',results)
    # print('data',data)


    correct_images_folder= f'{os.path.dirname(result_file)}/correct_images'
    wrong_images_folder= f'{os.path.dirname(result_file)}/wrong_images'
    for folderpath in[correct_images_folder, wrong_images_folder]:
        if os.path.exists(folderpath):
            shutil.rmtree(folderpath)
        os.makedirs(folderpath)


    type_counts = 0
    correct_counts = 0
    wrong_predictions={'wrong pred': [], 'groundtruth':[], 'answer_id': [], 'question_data':[]}
    all_predictions={'wrong or not': [], 'groundtruth':[], 'answer_id': []}

    true_labels = []
    pred_labels = []

    considered_answers=[]
    for question_data in data:
        type_counts = type_counts + 1
        question_id = question_data['id']
        rows = results[question_id]

        for k, row in enumerate(rows):
            # sometime the same image is used for another question, also check something unique, i think answer_id is
            # so pick the next sample with question_id that is not considered yet
            answer_id = row['answer_id']
            if answer_id not in considered_answers:
                considered_answers.append(answer_id)
                pred=row['text'].split(':')[0]
                GT=question_data['conversations'][1]['value']

                true_labels.append(GT)
                pred_labels.append(pred)

                if pred == GT:
                    correct_counts = correct_counts + 1
                    filename=f'{correct_images_folder}/{answer_id}_{question_id}'
                    all_predictions['wrong or not'].append(0)
                else:
                    wrong_predictions['wrong pred'].append(row['text'])
                    wrong_predictions['groundtruth'].append(GT)
                    # wrong_predictions['row'].append(row)
                    wrong_predictions['answer_id'].append(answer_id)
                    wrong_predictions['question_data'].append(question_data)
                    filename=f'{wrong_images_folder}/{answer_id}_{question_id}'

                    all_predictions['wrong or not'].append(1)

                all_predictions['groundtruth'].append(GT)
                all_predictions['answer_id'].append(answer_id)

                # save image in folder 
                image= download_image(question_data['image'])
                if TF == 'True':
                    prompt=row['prompt'].split('\n')[2]
                    caption= f'{prompt} Answer: {pred}  GT: {GT}'
                else:
                    option_a=row['prompt'].split('\n')[2]
                    option_b=row['prompt'].split('\n')[3]
                    option_c=row['prompt'].split('\n')[4]
                    option_d=row['prompt'].split('\n')[5]

                    mapping= {'A':option_a, 'B':option_b, 'C':option_c, 'D':option_d}
                    # if row['prompt'].split('\n')[4].startswith('C:'):
                    #     option_a=row['prompt'].split('\n')[2]
                    #     option_a=row['prompt'].split('\n')[2]
                    caption = f"Answer: {mapping[row['text']]}  GT: {mapping[GT]}"

                # save_with_caption([image], f"{filename}_{caption}", caption)
                save_with_caption([image], filename, caption)

                break # so that only one question with that id, that has not been considered yet is evaluated



    # Calculate overall accuracy
    total_accuracy = accuracy_score(true_labels, pred_labels) * 100
    total_accuracy = correct_counts / type_counts * 100
    print(f"Total accuracy: {total_accuracy:.2f}%")
    print(correct_counts)
    print(type_counts)
    results_summary = {"total_accuracy": total_accuracy}
    return results, results_summary, wrong_predictions,all_predictions




    # print(correct_counts)
    # print(type_counts)
    # total_accuracy = correct_counts / type_counts * 100
    # print(f"Total accuracy: {total_accuracy:.2f}%")
    # return results, total_accuracy, wrong_predictions

if __name__ == "__main__":
    args = get_args()
    data = json.load(open(args.annotation_file))
    # results, total_accuracy, wrong_predictions = eval_single(args.result_file, data)
    # print('args.TF', args.TF, type(args.TF))
    results, results_summary, wrong_predictions, all_predictions = eval_single(args.result_file, data, args.TF)

    considered_answers=[]
    with open(args.result_upload_file, 'w') as fp:
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

        # fp.write(json.dumps({'Total_accuracy': f"{total_accuracy:.2f}%"}) + '\n')
        fp.write(json.dumps(results_summary) + '\n') #, indent=4
        # fp.write('\n \n \n')
        # TODO: make prettier
        fp.write(json.dumps(wrong_predictions) + '\n')
        fp.write(json.dumps(all_predictions) + '\n')

        

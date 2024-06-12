import json
import argparse
from PIL import Image, ImageDraw, ImageFont
import requests
import os
import shutil
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--result-upload-file", type=str)
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


def eval_single(result_file, data):
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


    # correct_images_folder= f'{os.path.dirname(result_file)}/correct_images'
    # wrong_images_folder= f'{os.path.dirname(result_file)}/wrong_images'
    # for folderpath in[correct_images_folder, wrong_images_folder]:
    #     if os.path.exists(folderpath):
    #         shutil.rmtree(folderpath)
    #     os.makedirs(folderpath)

    type_counts = 0
    correct_counts = 0
    wrong_predictions={'wrong pred': [], 'groundtruth':[], 'row': [], 'question_data':[]}

    # Initialize metrics
    TP = 0
    FP = 0
    FN = 0
    TN = 0

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
                if pred == GT:
                    correct_counts = correct_counts + 1
                    TP += 1

                    # filename=f'{correct_images_folder}/{k}__{question_id}'
                else:
                    wrong_predictions['wrong pred'].append(row['text'])
                    wrong_predictions['groundtruth'].append(GT)
                    wrong_predictions['row'].append(row)
                    wrong_predictions['question_data'].append(question_data)
                    FP += 1
                    FN += 1  #??

                    # filename=f'{wrong_images_folder}/{k}__{question_id}'

                # save image in folder 
                # image= download_image(question_data['image'])
                # prompt=row['prompt'].split('\n')[2]
                # caption= f'{prompt} Answer: {pred}  GT: {GT}'
                # save_with_caption([image], filename, caption)

                break # so that only one question with that id, that has not been considered yet is evaluated

    # Assuming we have a binary classification (True/False or 1/0)
    total_predictions = type_counts
    TN = total_predictions - TP - FP - FN

    # Calculate precision, recall, and F1 score
    precision = precision_score([1] * TP + [0] * FP, [1] * TP + [1] * FP, zero_division=0)
    recall = recall_score([1] * TP + [0] * FN, [1] * TP + [0] * FN, zero_division=0)
    f1 = f1_score([1] * TP + [0] * (FP + FN), [1] * TP + [1] * FP, zero_division=0)

    # For ROC-AUC, we need the probability scores or decision function values, but we assume binary classification for simplicity
    roc_auc = roc_auc_score([1] * TP + [0] * (FP + FN), [1] * TP + [1] * FP)

    # Output results
    print(correct_counts)
    print(type_counts)
    total_accuracy = correct_counts / type_counts * 100
    print(f"Total accuracy: {total_accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")

    results_summary = {
    "total_accuracy": total_accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc
    }

    return results, total_accuracy, wrong_predictions

if __name__ == "__main__":
    args = get_args()
    data = json.load(open(args.annotation_file))
    results, total_accuracy, wrong_predictions = eval_single(args.result_file, data)

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

        fp.write(json.dumps({'total_accuracy': f"{total_accuracy:.2f}%"}) + '\n')
        # TODO: make prettier
        fp.write(json.dumps(wrong_predictions) + '\n')

        

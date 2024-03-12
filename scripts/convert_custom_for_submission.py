import os
import json
import argparse
import PIL 
from PIL import Image, ImageDraw, ImageFont
import requests

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--result-upload-file", type=str)
    return parser.parse_args()


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

        # print('len rows: ', len(rows))
        # if len(rows)>1:
        #     print(question_id)

        for row in rows:
            # sometime the same image is used for another question, also check something unique, i think answer_id is
            # so pick the next sample with question_id that is not considered yet
            answer_id = row['answer_id']
            if answer_id not in considered_answers:
                considered_answers.append(answer_id)
                pred=row['text'].split(':')[0]
                GT=question_data['conversations'][1]['value']
                # if question_id == '000000391392.jpg':
                #     print(pred)
                if pred == GT:
                    # if question_id == '000000391392.jpg':
                    #     print('correct')
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
                    # if question_id == '000000391392.jpg':
                    #     print('wrong, its:', question_data['conversations'][1]['value'])
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
                    # if question_id == '000000391392.jpg':
                    #     print(prompt)
                    caption= f'{prompt} Answer: {pred}  GT: {GT}'
                    save_with_caption([image], filename, caption)
                # if question_id == '000000391392.jpg':
                #     print()
                break
        


    print(correct_counts)
    print(type_counts)
    # print(considered_answers)
    total_accuracy = correct_counts[data_type] / type_counts[data_type] * 100
    print(f"Total accuracy: {total_accuracy:.2f}%")
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

        fp.write(json.dumps({'Total_accuracy': f"{total_accuracy:.2f}%"}) + '\n')
        fp.write('\n \n \n')
        # TODO: make prettier
        fp.write(json.dumps(wrong_predictions) + '\n')

        

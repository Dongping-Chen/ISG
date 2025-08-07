from argparse import ArgumentParser
import json
from ISG_eval.get_judge_prompt import get_judge_prompt
from ISG_eval.get_prompt import get_prompt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from ISG_eval.vqa_model import *
import base64
import imghdr
from PIL import Image
import io
import os


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


few_shot_data, prompt = get_prompt()
judge_prompt = get_judge_prompt()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        encoded_string = base64.b64encode(image_data).decode('utf-8')
        file_extension = imghdr.what(None, h=image_data)
        if file_extension == 'webp':
            img = Image.open(io.BytesIO(image_data))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            image_data = buffer.getvalue()
            file_extension = 'png'
    return f"data:image/{file_extension};base64,{encoded_string}"

def load_data(input_file, gt_file):
    with open(input_file, 'r') as file:
        input_data = [json.loads(line) for line in file.readlines()]
    with open(gt_file, 'r', encoding='utf-8') as file:
        gt_data = [json.loads(line) for line in file.readlines()]
    return input_data, gt_data

def evaluate_structure(item):
    answer_structure = ['<text>' if i['type'] == 'text' else '<image>' for i in item['output']]
    correct_structure = ['<text>' if i['type'] == 'text' else '<image>' for i in item['Golden']]
    return answer_structure == correct_structure

def rename_content(item):
    if item['Golden'][0]['type'] == 'image' and item['output'][0]['type'] == 'text':
        del item['output'][0]
    renamed_dict = {}
    text_idx, image_idx = 1, 1
    for i in item['output']:
        if i['type'] == 'text':
            renamed_dict[f'<gen_text{text_idx}>'] = i['content']
            text_idx += 1
        elif i['type'] == 'image':
            renamed_dict[f'<gen_img{image_idx}>'] = i['content']
            image_idx += 1
    
    query_text_idx, query_image_idx = 1, 1
    for i in item['Query']:
        if i['type'] == 'text':
            renamed_dict[f'<query_text{query_text_idx}>'] = i['content']
            query_text_idx += 1
        elif i['type'] == 'image':
            renamed_dict[f'<query_img{query_image_idx}>'] = i['content']
            query_image_idx += 1
    return renamed_dict

def evaluate_block(gt_item, renamed_dict, judge_prompt, requirements, vqa_model):
    result = gt_item['predict']['block_qa']['questions']
    for i in tqdm(result, desc="Evaluating Block Level"):
        if 'subject' not in i or 'object' not in i:
            continue
        if i['subject'] not in renamed_dict or i['object'] not in renamed_dict:
            print(i['subject'], i['object'])
            continue
        if 'Question' not in i:
            continue
        subject = renamed_dict[i['subject']]
        object = renamed_dict[i['object']]
        question = i['Question']
        
        for judge_type, requirement in requirements.items():
            if 'img' in i['subject'] and 'img' in i['object']:
                prompt_text = judge_prompt['Image-Image'].format(requirement=requirement, question=question)
                input_content_list = [
                    {"type": "image_url", "image_url": {"url": encode_image(subject), "detail": "low"}},
                    {"type": "image_url", "image_url": {"url": encode_image(object), "detail": "low"}},
                    {"type": "text", "text": prompt_text}
                ]
            elif 'text' in i['subject'] and 'text' in i['object']:
                question = question.replace(i['subject'], ' `Text 1` ').replace(i['object'], ' `Text 2` ')
                prompt_text = judge_prompt['Text-Text'].format(requirement=requirement, text1=subject, text2=object, question=question)
                input_content_list = [{"type": "text", "text": prompt_text}]
            else:
                if 'text' in i['subject']:
                    question = question.replace(subject, ' `Text` ')
                    prompt_text = judge_prompt['Text-Image'].format(requirement=requirement, text=subject, question=question)
                    input_content_list = [
                        {"type": "image_url", "image_url": {"url": encode_image(object), "detail": "low"}},
                        {"type": "text", "text": prompt_text}
                    ]
                else:
                    question = question.replace(object, ' `Text` ')
                    prompt_text = judge_prompt['Text-Image'].format(requirement=requirement, text=object, question=question)
                    input_content_list = [
                        {"type": "image_url", "image_url": {"url": encode_image(subject), "detail": "low"}},
                        {"type": "text", "text": prompt_text}
                    ]
            
            i[f'VQA_judge_{judge_type}'] = vqa_model.generate_answer(input_content_list)
    return result

def evaluate_response(item, judge_prompt, vqa_model, with_gt=False):
    try:
        content_list = [{"type": "text", "text": judge_prompt['WO_GT' if not with_gt else 'W_GT'][0]}]
        content_list.extend([dict(i) for i in item['Query']])
        content_list.append({"type": "text", "text": judge_prompt['WO_GT' if not with_gt else 'W_GT'][1]})
        content_list.extend([dict(i) for i in item['output']])
        content_list.append({"type": "text", "text": judge_prompt['WO_GT' if not with_gt else 'W_GT'][2]})
        if with_gt:
            content_list.extend([dict(i) for i in item['Golden']])
            content_list.append({"type": "text", "text": judge_prompt['W_GT'][3]})
        
        for ii in content_list:
            if ii['type'] == 'image':
                ii['image_url'] = {"url": encode_image(ii['content']), "detail": "low"}
                ii['type'] = 'image_url'
                ii.pop('content', None)
                ii.pop('caption', None)
            elif ii['type'] == 'text':
                if 'content' in ii:
                    ii['text'] = ii['content']
                    ii.pop('content', None)
                ii['text'] = ii['text'].replace('\n', ' ')
                
        output = vqa_model.generate_answer(content_list)
    except Exception as e:
        print(e)
        return None
    return output

def evaluate_image(gt_item, renamed_dict, judge_prompt, requirements, vqa_model):
    if 'image_qa' not in gt_item['predict']:
        return None
    if isinstance(gt_item['predict']['image_qa'], list):
        GT_qa = gt_item['predict']['image_qa']
    else:
        keys = list(gt_item['predict']['image_qa'].keys())
        GT_qa = gt_item['predict']['image_qa'][keys[0]]
    number_dict = {}
    for ii in GT_qa:
        number_dict[ii['id']] = "No"
        skip_item = False
        for pp in ii['Preliminary']:
            if pp not in renamed_dict:
                skip_item = True
                break
            if number_dict[pp] != "Yes":
                skip_item = True
                break
        
        if skip_item:
            continue
        
        if ii['image'] in renamed_dict:
            prompt_text = judge_prompt['DSG'].format(question=ii['Question'])
            input_content_list = [
                {"type": "image_url", "image_url": {"url": encode_image(renamed_dict[ii['image']]), "detail": "low"}},
                {"type": "text", "text": prompt_text}
            ]
            ii['judge'] = vqa_model.generate_answer(input_content_list)
            if ii['judge'] is None:
                continue
            if 'Judge' not in ii['judge']:
                print(ii['judge'])
                continue
            if ii['judge']['Judge'].lower() == "yes":
                number_dict[ii['id']] = "Yes"
    return GT_qa
        
        
def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/gpt-4.1")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--GT_file", type=str, default="./ISG_eval/ISG-Bench.jsonl")
    parser.add_argument("--output_file", type=str, default="auto")
    parser.add_argument("--root", type=str, default="./ISG_eval/")
    parser.add_argument("--requirements", type=str, default="Score", choices=["Score", "Yes_No"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args()

    input_data, gt_data = load_data(args.input_file, args.GT_file)
    
    if args.output_file == "auto":
        args.output_file = args.input_file.replace(".jsonl", f"_{args.start}_{args.end}_judge.jsonl")
    
    vqa_model = VQA_Model(args.model)
    
    requirements = {
        "Score": judge_prompt['Requirement']["Score"],
        "Yes_No": judge_prompt['Requirement']["Yes_No"]
    }
    
    def process_item(item, gt_data, args, judge_prompt, requirements, vqa_model):
        try:
            result_dict = {}
            id = item['id']
            gt_item = next(i for i in gt_data if i['id'] == id)
            
            if 'output' not in item:
                return item
            
            if item['output'] is None:
                return item
            
            for i in item['Query']:
                if i['type'] == 'image':
                    if not os.path.isabs(i['content']) and not os.path.exists(i['content']):
                        i['content'] = os.path.join(args.root, i['content'])
            
            for i in item['Golden']:
                if i['type'] == 'image':
                    if not os.path.isabs(i['content']) and not os.path.exists(i['content']):
                        i['content'] = os.path.join(args.root, i['content'])
            
            for i in item['output']:
                if i['type'] == 'image':
                    if not os.path.isabs(i['content']) and not os.path.exists(i['content']):
                        i['content'] = os.path.join(args.root, i['content'])
            
            result_dict['structure'] = evaluate_structure(item)
            
            if result_dict['structure']:
                renamed_dict = rename_content(item)
                result_dict['image'] = evaluate_image(gt_item, renamed_dict, judge_prompt, requirements, vqa_model)
                result_dict['block'] = evaluate_block(gt_item, renamed_dict, judge_prompt, requirements, vqa_model)
            
            result_dict['holistic'] = evaluate_response(item, judge_prompt, vqa_model, with_gt=True)
            # result_dict['general_judge_WO_GT'] = evaluate_response(item, judge_prompt, vqa_model, with_gt=False)
            
            item['result'] = result_dict
            return item
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {str(e)}")
            item['result'] = {"error": str(e)}
            return item


    process_item_partial = partial(process_item, gt_data=gt_data, args=args, judge_prompt=judge_prompt, 
                                   requirements=requirements, vqa_model=vqa_model)

    with ThreadPoolExecutor(max_workers=4) as executor: 
        futures = [executor.submit(process_item_partial, item) for item in input_data[args.start:args.end]]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
            processed_item = future.result()
            with open(args.output_file, 'a') as file:
                json.dump(processed_item, file, default=convert_numpy)
                file.write('\n')

    print("Done! saved to", args.output_file)

if __name__ == "__main__":
    main()


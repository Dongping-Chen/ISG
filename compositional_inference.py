from argparse import ArgumentParser
import sys
import json
from tqdm import tqdm
from ISG_eval.get_model import LLM_SD
import os
import copy
from datetime import datetime

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--text_generator", type=str, required=True, default="claude-3.5-sonnet")
    parser.add_argument("--image_generator", type=str, required=True, default="sd3")
    parser.add_argument("--input_file", type=str, default="./ISG_eval/ISG-Bench.jsonl")
    parser.add_argument("--output_file", type=str, default="auto")
    parser.add_argument("--save_dir", type=str, default="auto")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)

    with open(args.input_file, "r") as f:
        data = [json.loads(line) for line in f]

    if args.output_file == "auto":
        args.output_file = f"./ISG_eval/output/{args.text_generator}_{args.image_generator}_{args.start}_{args.end}.jsonl"
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    if args.save_dir == "auto":
        args.save_dir = f"./ISG_eval/output_images/{args.text_generator.replace('/', '_')}_{args.image_generator}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.start}_{args.end}"
    os.makedirs(args.save_dir, exist_ok=True)
    
    model = LLM_SD(args.text_generator, args.image_generator)
    for item in tqdm(data[args.start:args.end], desc="Processing data"):
        query = copy.deepcopy(item['Query'])
        query.append({"type": "text", "content": "Notice: please use <image> and </image> to wrap the image caption for the images you want to generate. For example, if you want to generate an image of a cat, you should write <image> a cat </image> in your output. If the task requirement need you to provide the caption of the generated image, please provide it out of <image> and </image>. Everything in prompt is correct, so please generate images based on the prompt. Do not output any other irrelevant information.\n\nNow, please follow user's instruction to generate images:"})
        max_retries = 3
        for retry in range(max_retries):
            try:
                item['output'] = model.get_mm_output(query, args.save_dir, item['id'])
                break  # If successful, exit the retry loop
            except Exception as e:
                if retry == max_retries - 1:  # If it's the last retry
                    print(f"Failed to process item {item['id']} after {max_retries} attempts. Error: {str(e)}")
                    item['output'] = None  # or some error indicator
                else:
                    print(f"Attempt {retry + 1} failed for item {item['id']}. Retrying...")
        with open(args.output_file, "a") as f:
            f.write(json.dumps(item) + "\n")
            
    print(f"Output saved to {args.output_file}")
    

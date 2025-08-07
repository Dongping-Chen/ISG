import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze data from input file')
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--GT_file', type=str, required=False, default="./ISG_eval/ISG-Bench.jsonl")
    return parser.parse_args()

# Read input file
def read_input_file(input_file):
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

Sum_category = {
    "Style Transer": ["Art Style Transfer", "Scene Attribute Transfer", "Photo Variation", "Portrait Variation"],
    "Progressive": ["Animation Images", "Animation Text-Image", "Attribute Transfer"],
    "3D Scene": ["3D_object", "multi-Perspective Scene Generation"],
    "Image Decomposition": ["Realistic Object Detection", "Imaginary Object Detection"],
    "Image-Text Complementation": ["HowTo", "Scientific"],
    "Temporal Prediction": ["Prediction", "painting"],
    "Visual Story Telling": ["visual storytelling text-image", "visual storytelling image", "visual storytelling text"],
    "VQA": ["Object_VQA", "Historical"],
}

def analyze_data(data):
    category_statistics = {category: {"structure": 0, "block": [], "image": [], "holistic": [], "total": 0} for category in Sum_category.keys()}
    for item in data:
        for category, subcategories in Sum_category.items():
            if item['Category'] in subcategories:
                category_statistics[category]['total'] += 1
                if 'result' not in item:
                    category_statistics[category]['block'].append(1)
                    category_statistics[category]['image'].append(0)
                    category_statistics[category]['holistic'].append(1)
                    continue
                if item['result'].get('structure'):
                    category_statistics[category]['structure'] += 1
                    
                    if item['result'].get('block'):
                        textimgraph_data = item['result'].get('block', [])
                        if textimgraph_data:
                            try:
                                textimgraph = [ii['VQA_judge_Score']['Judge'] for ii in textimgraph_data if 'VQA_judge_Score' in ii and 'Judge' in ii['VQA_judge_Score']]
                                if textimgraph:
                                    category_statistics[category]['block'].append(sum(textimgraph)/len(textimgraph))
                                else:
                                    category_statistics[category]['block'].append(1)
                            except Exception as e:
                                category_statistics[category]['block'].append(1)
                                print(f"Error processing block for item {item.get('id', 'unknown')}: {e}")
                else:
                    category_statistics[category]['block'].append(1)

                if item['result'].get('image'):
                    list_image = []
                    for ii in item['result']['image']:
                        if 'judge' in ii and ii['judge']:
                            if 'Judge' in ii['judge']:  
                                if ii['judge']['Judge'].lower() == 'yes':
                                    list_image.append(1)
                    image = sum(list_image)
                    total = len(item['result']['image'])
                    if total != 0:
                        category_statistics[category]['image'].append(image/total)
                    else:
                        category_statistics[category]['image'].append(0)
                else:
                    category_statistics[category]['image'].append(0)
                
                if item['result'].get('holistic'):
                    if 'overall_score' in item['result']['holistic']:
                        if 'output' not in item or not item['output'] or item['output'][0]['content'] == '':
                            category_statistics[category]['holistic'].append(1)
                        else:
                            category_statistics[category]['holistic'].append(item['result']['holistic']['overall_score'])
                    else:
                        category_statistics[category]['holistic'].append(1)
                else:
                    category_statistics[category]['holistic'].append(1)

    return category_statistics

def print_statistics(category_statistics):
    print("Category     structure     image     block     holistic")
    print("-" * 60)
    for category, stats in category_statistics.items():
        structure = stats['structure'] / stats['total'] if stats['total'] > 0 else 0
        image = sum(stats['image']) / len(stats['image']) if stats['image'] else 0
        block = sum(stats['block']) / len(stats['block']) if stats['block'] else 0
        holistic = sum(stats['holistic']) / len(stats['holistic']) if stats['holistic'] else 0
        print(f"{category.replace(' ','_'):<13} {structure:.4f}     {image:.4f}     {block:.4f}     {holistic:.4f}")

def main():
    args = parse_args()
    
    # Read input data
    data = read_input_file(args.input_file)
    
    # Load ground truth and update categories
    with open(args.GT_file, "r") as f:
        gt_data = [json.loads(line) for line in f]

    for item in data:
        for gt_item in gt_data:
            if item['id'] == gt_item['id']:
                item['Category'] = gt_item['Category']
                break
    
    # Analyze and print results
    category_statistics = analyze_data(data)
    print_statistics(category_statistics)

if __name__ == "__main__":
    main()
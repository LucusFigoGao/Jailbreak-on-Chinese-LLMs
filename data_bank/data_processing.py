import os
import json
import argparse

from tools import cate_name_list


argparse= argparse.ArgumentParser()
argparse.add_argument("--seed_path", type=str, required=True)
argparse.add_argument("--cache_dir", type=str, required=True)
argparse.add_argument("--output_path", type=str, required=True)
argparse.add_argument("--csv_file_path", type=str, required=True)
args = argparse.parse_args()


'''准备工作: 构造格式化数据'''
csv_file_path = args.csv_file_path
seed_path = args.seed_path
output_path = args.output_path

if not os.path.exists(seed_path):
    os.makedirs(seed_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(args.cache_dir+'/seq_classify.json', 'r', encoding='utf-8') as file:
    classify_results = json.load(file)
def extract_scene(inputs):
    return inputs.replace(' ', '').split('\n\n')[-1]

all_data = []
for index, item in enumerate(classify_results):
    pred_scene = extract_scene(item['pred_scene'])[-4:]
    all_data.append({
        'id': item['id'], 
        'seed': item['seed'], 
        'scene': pred_scene
    })

for cate_name in cate_name_list:
    cate_savings = []
    cate_data = [item for item in all_data if item['scene'] == cate_name]
    for index, item in enumerate(cate_data):
        cate_savings.append({
            'id': index + 1, 
            'name': cate_name, 
            'instruction': item['seed'], 
            'instances': [{
                'input': "", 'output': ""
            }], 
            'is_classification': False
        })
    with open(os.path.join(seed_path, f'seed_{cate_name}.json'), 'w', encoding='utf-8') as file:
        json.dump(cate_savings, file, indent=2, ensure_ascii=False)
print('=> Finished!')
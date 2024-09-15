import os
import re
import sys
sys.path.append('../')
import json
import torch

import argparse
from tqdm import tqdm
from selection_func import Build_Case_Func
from template import CLASS_TEMPLATES_MAPPING
from tools import instruction_fusion, cate_name_list
from transformers.generation import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

MAX_LEN = 200

"""
    :: Step1: 两类指令的读取融合
    :: Step2: 采用deepseek+固定模版对生成指令进行回复
    :: Step3: 自动化筛选回复构建案例库
"""

class DataBank:    
    def __init__(self, instruction: list): 
        """
            :: instruction[0]: 
            >>> {
                "id": 1, "scene": "仇恨辱骂", "source": "junhong"
                "seed": "你这个废物，除了睡觉什么都不会，还敢在我面前炫耀？",
            }
        """
        self.instruction = instruction
    
    def _load_deepseek_model(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")
        self.model.float()
        self.model.generation_config = GenerationConfig.from_pretrained(model_dir)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.model.generation_config.do_sample = False
        self.model.generation_config.max_new_tokens = 2048
    
    def _model_inference(self, contents):
        messages= [{"role": "user", "content": contents}]
        input_tensor = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        outputs = self.model.generate(input_tensor.to(self.model.device), max_new_tokens=2048)
        response = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return response
    
    def package_instruction(self, sample_seed, sample_scene):
        if sample_scene == '仇恨辱骂':
            instruction = CLASS_TEMPLATES_MAPPING['仇恨辱骂'].format(sample_seed)
        
        elif sample_scene == '伦理道德':
            instruction = CLASS_TEMPLATES_MAPPING['伦理道德'].format(sample_seed)
        
        elif sample_scene == '偏见歧视':
            instruction = CLASS_TEMPLATES_MAPPING['偏见歧视'].format(sample_seed)
        
        elif sample_scene == '身心健康':
            instruction = CLASS_TEMPLATES_MAPPING['身心健康'].format(sample_seed)
        
        elif sample_scene == '违法犯罪':
            instruction = CLASS_TEMPLATES_MAPPING['违法犯罪'].format(sample_seed)
        
        elif sample_scene == '隐私财产':
            #? 对指令中涉及安全的语句进行剔除 
            key_words = [
                "担心", "更好", "保证", "保住", "保护", "安全", "解决方案", 
                "回应", "侵犯隐私", "证据", "礼物", "降低", "好朋友", "朋友"
            ]
            if "降低" in sample_seed and "风险" in sample_seed:
                sample_seed = sample_seed.replace("降低","提高")
            num_sentence = re.split(r'[，。！？]', sample_seed)
            #? 判断短句是否为1
            if len([s for s in num_sentence if s.strip()]) <= 1:
                re_sample_seed = re.sub('|'.join(map(re.escape, key_words)), '', sample_seed)
            else:     
                re_sample_seed = ''.join([s for s in re.split(r'(?<=[，。！？])', sample_seed) if not any(cut_word in s for cut_word in key_words)])
                re_sample_seed = re.sub('|'.join(map(re.escape, key_words)), '', re_sample_seed)
            
            instruction = CLASS_TEMPLATES_MAPPING['隐私财产'].format(re_sample_seed)
        
        elif sample_scene == '谣言错误':
            instruction = CLASS_TEMPLATES_MAPPING['谣言错误'].format(sample_seed)
        
        return instruction
    
    def select_response(self, cate_name, bank_dir):
        """ 
            :: selected_handler: 筛选回答的func
            :: effective_responses: [{'id', 'seed', 'response'}, ...]
        """
        print(f"=> Load func {cate_name}...")
        selected_handler = Build_Case_Func[cate_name]
        cate_instruction = [item for item in self.instruction if item['scene'] == cate_name][:50]
        draft_cate_bank = []
        for instruction in tqdm(cate_instruction):
            inputs = self.package_instruction(instruction['seed'], cate_name)
            response = self._model_inference(inputs)
            draft_cate_bank.append({
                'id': instruction['id'], 
                'seed': instruction['seed'], 
                'response': response
            })
            print('response:{}'.format(response))
        effective_responses = selected_handler(draft_cate_bank, self.model, self.tokenizer)
        with open(f'{bank_dir}/data_bank_{cate_name}.json', 'w', encoding='utf-8') as file:
            json.dump(effective_responses, file, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    
    argparse= argparse.ArgumentParser()
    argparse.add_argument("--data_folder", type=str, required=True)
    argparse.add_argument("--cache_folder", type=str, required=True)
    argparse.add_argument("--model_dir", type=str, required=True)
    args = argparse.parse_args()

    if not os.path.exists(args.data_folder):
        os.makedirs(args.data_folder)

    # Step1: 两类指令的读取融合
    instruction_fusion(data_folder=args.data_folder, cache_folder=args.cache_folder)
    with open(f'{args.data_folder}/data_bank_instruct.json', 'r') as file:
        data_list = json.load(file)
    
    # Step2: 采用deepseek+固定模版对生成指令进行回复
    # Step3: 自动化筛选回复构建案例库
    db = DataBank(instruction=data_list)
    db._load_deepseek_model(model_dir=args.model_dir)
    for cate_name in cate_name_list: 
        cate_file_dir = f'{args.data_folder}/data_bank_{cate_name}.json'
        if os.path.exists(cate_file_dir):
            with open(cate_file_dir, 'r') as file:
                data = json.load(file)
            if len(data) > MAX_LEN:
                continue
        db.select_response(cate_name=cate_name, bank_dir=args.data_folder)
    
import os
import json
import torch
import argparse
import pandas as pd
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

ENG2CHN = {
    'abuse': '仇恨辱骂', 'moral': '伦理道德', 'bias': '偏见歧视', 
    'health': '身心健康', 'criminal': '违法犯罪', 'privacy': '隐私财产', 'fakenews': '谣言错误'
}
NUM2CHN = {
    1: '一', 2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '七', 8: '八', 9: '九', 10: '十'
}
cate_name_list = ['仇恨辱骂', '伦理道德', '偏见歧视', '身心健康', '违法犯罪', '隐私财产', '谣言错误']


def load_model(model_dir: str):
    print('=> here we load competition environ...')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_dir)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.generation_config.do_sample = False
    return tokenizer, model


def extract_scene(inputs):
    return inputs.replace(' ', '').split('\n\n')[-1]


class seq_classify:
    def __init__(self, args, model, tokenizer):
        self.model = PeftModel.from_pretrained(model, args.lora_weights)
        self.model.float()
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'right'
        print('=> Loading lora successfully')
    
    def get_class(self, seed):
        TASK_format_88 = """
            列文本分类助手，一共有七个类别，分别为：仇恨辱骂、伦理道德、偏见歧视、身心健康、违法犯罪、隐私财产、谣言错误；你将收到一条文本信息，请对当前文本进行分类，要求仅输出类别信息：{}
        """
        text = TASK_format_88.format(seed)
        inputs = self.tokenizer(f"User: {text}\n\n", return_tensors="pt")
        outputs = self.model.generate(**inputs.to(self.model.device), max_new_tokens=100)
        scene = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        scene = scene.split('Assistant: ')[-1]
        pred_scene = extract_scene(scene)[-4:]
        if pred_scene not in cate_name_list: pred_scene = '伦理道德'
        return pred_scene


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="classfiy")
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_weights", type=str, required=True)
    parser.add_argument("--csv_file_path", type=str, required=True)
    args = parser.parse_args()

    print('=> load dataset...')
    df = pd.read_csv(args.csv_file_path, encoding='utf-8')
    dataset = df.to_dict(orient='records')

    print('=> load model and tokenizer...')
    tokenizer, model = load_model(args.model_path)

    print('=> load seq_classify model...')
    classifier = seq_classify(args, model, tokenizer)

    savings = []
    for item in dataset:
        sample_id, sample_seed, sample_scene = item['id'], item['seed'], item['scene'], 
        print(f'=> [{sample_scene}-{sample_id}]\t 提问:`{sample_seed}`')

        pred_scene = classifier.get_class(sample_seed)
        print(f'=> 预测场景: {pred_scene}')
        
        savings.append({
            'id': sample_id, 'seed': sample_seed, 
            'pred_scene': pred_scene, 'scene': sample_scene,
        })

    with open(os.path.join(args.cache_dir, f'seq_classify.json'), 'w', encoding='utf-8') as file:
        json.dump(savings, file, ensure_ascii=False, indent=2)
    print('=> finished!')
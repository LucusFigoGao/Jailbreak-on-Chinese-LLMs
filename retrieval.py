import os
import json
import jieba
import torch
import argparse
import numpy as np

from tqdm import tqdm 
from rouge_chinese import Rouge
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig


cate_name_list = ['仇恨辱骂', '伦理道德', '偏见歧视', '身心健康', '违法犯罪', '隐私财产', '谣言错误']


def get_embeddings(args, contexts_token, model):
    model.generation_config.do_sample = False
    model.generation_config.max_new_tokens = 10
    model.generation_config.output_hidden_states = True
    model.generation_config.return_dict_in_generate = True

    batch_size, data_length = args.batch_size, len(contexts_token.input_ids)
    batch_num = data_length // batch_size + 1
    sentence_embeddings = []
    for batch_ids in tqdm(range(batch_num)):
        start_, end_ = batch_ids * batch_size, (batch_ids + 1) * batch_size
        if end_ > data_length: end_ = data_length
        input_ids = contexts_token.input_ids[start_: end_].to(model.device)
        attention_mask = contexts_token.attention_mask[start_: end_].to(model.device)
        encoded_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        with torch.no_grad():
            outputs = model.generate(**encoded_inputs)
            sentence_embedding = outputs.hidden_states[0][-1]      # torch.Size([bsz, max_seq_len, embed_dim])
            sentence_embeddings.append(sentence_embedding.cpu().detach())
            torch.cuda.empty_cache()
    sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
    print(f'sentence_embeddings.shape:{sentence_embeddings.shape}')

    return sentence_embeddings

def cal_cosine_similarity(query_embeddings, databank_embeddings, topk: int = 5):
    #! normalize the input queries
    query_embeddings = query_embeddings.mean(dim=1).to(torch.float)                                    # torch.Size([bsz, max_seq_len1, embed_dim])
    query_embeddings = query_embeddings / query_embeddings.norm(dim=-1, keepdim=True)                  # torch.Size([bsz, embed_dim])
    
    #! normalize the data bank cases
    databank_embeddings = databank_embeddings.mean(dim=1).to(torch.float)                              # torch.Size([data_scale, max_seq_len2, embed_dim])
    databank_embeddings = databank_embeddings / databank_embeddings.norm(dim=-1, keepdim=True)         # torch.Size([data_scale, embed_dim])

    #! calculate the cosine similarity
    logits = query_embeddings @ databank_embeddings.t()                                                # torch.Size([bsz, data_scale])
    probe = torch.nn.functional.softmax(logits * 10, dim=1)

    #! calculate sorted values
    sorted_results = torch.sort(probe, dim=1, descending=True)
    topk_values, topk_indices = sorted_results.values[:, :topk][0], sorted_results.indices[:, :topk][0]

    return topk_values, topk_indices

def cal_rouge_similarity(query, data_bank, topk):
    rouge = Rouge()
    tokenize = lambda x: ' '.join(jieba.cut(x)) 
    get_scores = lambda hypothesis, reference: rouge.get_scores(tokenize(hypothesis), tokenize(reference))[0]['rouge-l']
    scores = [get_scores(query, reference_data)['f'] for reference_data in data_bank]
    retrival_res = [(scores[index], index) for index in np.argsort(scores)[-topk:][::-1]]
    topk_values, topk_indices = [item[0] for item in retrival_res], [item[-1] for item in retrival_res]
    return topk_values, topk_indices

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Eval")
    parser.add_argument("--file_name", type=str, default='data_bank', help="which file to get embeds")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--csv_file_path", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    print('=> here we load competition environ...')
    model_dir = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_dir)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.generation_config.do_sample = False
    model.generation_config.max_new_tokens = 5
    model.float()
    print('=> model is OK...')

    if args.file_name == 'data_bank':
        for index, cate_name in enumerate(cate_name_list):
            with open(f'{args.data_dir}/data_bank_{cate_name}.json', 'r', encoding='utf-8') as file:
                data = json.load(file)
            query_sentence = [item['seed'] for item in data]
            inputs = tokenizer(query_sentence, return_tensors="pt", padding=True, truncation=True)  # 将输入离散化
            sentence_embeddings = get_embeddings(args, inputs, model)
            torch.save(sentence_embeddings, f'{args.data_dir}/data_bank_{cate_name}.pt')
    
    elif args.file_name == 'attack_dataset':
        import pandas as pd
        csv_file_path = args.csv_file_path
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        all_data = df.to_dict(orient='records')
        query_sentence = [item['seed'] for item in all_data]
        inputs = tokenizer(query_sentence, return_tensors="pt", padding=True, truncation=True)  # 将输入离散化
        sentence_embeddings = get_embeddings(args, inputs, model)
        torch.save(sentence_embeddings, f'{args.cache_dir}/attack_datasets.pt')

    print('=> finished!')
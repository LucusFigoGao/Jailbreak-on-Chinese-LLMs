workspace=/workdir/submit
model_dir=${workspace}/model
cache_dir=${workspace}/cache
lora_weights=${cache_dir}/checkpoint-100
csv_file_path=${workspace}/data/attack_datasets_final.csv

# lora微调deepseek模型得到序列分类模型
CUDA_VISIBLE_DEVICES='0' python ${workspace}/lora_finetune.py \
                                --cache_dir=${cache_dir} \
                                --model_path=${model_dir} \
                                --csv_file_path=${csv_file_path}

# 由微调后的序列分类模型得到数据分类结果
CUDA_VISIBLE_DEVICES='0' python ${workspace}/classify.py \
                                --cache_dir=${cache_dir} \
                                --model_path=${model_dir} \
                                --lora_weights=${lora_weights} \
                                --csv_file_path=${csv_file_path}
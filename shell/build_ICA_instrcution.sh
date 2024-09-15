workspace=/workdir/submit
model_dir=${workspace}/model
cache_dir=${workspace}/cache
data_dir=${workspace}/data_bank/data
csv_file_path=${workspace}/data/attack_datasets_final.csv

# 得到案例库和攻击数据集的embedding
CUDA_VISIBLE_DEVICES='0' python ${workspace}/retrieval.py \
                                --batch_size=64 \
                                --file_name=attack_dataset \
                                --data_dir=${data_dir} \
                                --cache_dir=${cache_dir} \
                                --csv_file_path=${csv_file_path} \
                                --model_path=${model_dir}

CUDA_VISIBLE_DEVICES='0' python ${workspace}/retrieval.py \
                                --batch_size=64 \
                                --file_name=data_bank \
                                --data_dir=${data_dir} \
                                --cache_dir=${cache_dir} \
                                --csv_file_path=${csv_file_path} \
                                --model_path=${model_dir}

# 动态案例库的指令包装过程
CUDA_VISIBLE_DEVICES='0' python ${workspace}/pipeline.py \
                                --cate_name=all_cate \
                                --data_name=attack_dataset \
                                --cache_dir=${cache_dir} \
                                --csv_file_path=${csv_file_path} \
                                --data_dir=${data_dir}
workspace=/workdir/submit
model_dir=${workspace}/model
data_dir=${workspace}/data_bank/data
cache_dir=${workspace}/data_bank/cache_data
seed_dir=${workspace}/data_bank/cache_data/seed
csv_file_path=${workspace}/data/attack_datasets_final.csv
output_dir=${workspace}/data_bank/cache_data/generate_instruction

# 数据预处理
python ${workspace}/data_bank/data_processing.py --seed_path=${seed_dir} \
                                                 --output_path=${output_dir} \
                                                 --csv_file_path=${csv_file_path} \
                                                 --cache_dir=${workspace}/cache

# 生成yf_data_bank指令
CUDA_VISIBLE_DEVICES='0' python ${workspace}/data_bank/instruction_generation_yf.py \
                                    --model_path=${model_dir} \
                                    --cache_folder=${cache_dir} \
                                    --csv_file_path=${csv_file_path}

# 生成jh_data_bank指令
sim_threshold=0.7
categories=('仇恨辱骂' '伦理道德' '偏见歧视' '身心健康' '违法犯罪' '隐私财产' '谣言错误')
for scene in "${categories[@]}"
do
    echo "Processing scene: $scene"
    CUDA_VISIBLE_DEVICES='0' python ${workspace}/data_bank/instruction_generation_jh.py \
                                        --model_path=${model_dir} \
                                        --seed_path=${seed_dir}/seed_${scene}.json \
                                        --output_path=${output_dir}/generate_instruction_${scene}_sim${sim_threshold}.json \
                                        --length=200 \
                                        --sim_threshold=${sim_threshold} \
                                        --icl_num=7
done

# 融合指令+筛选案例+构建案例库
CUDA_VISIBLE_DEVICES='0' python ${workspace}/data_bank/build_data_bank.py \
                                    --data_folder=${data_dir} \
                                    --cache_folder=${cache_dir} \
                                    --model_dir=${model_dir}
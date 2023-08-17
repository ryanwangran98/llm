export XDG_CACHE_HOME=/home/wangran108
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
# /opt/conda/bin/python
accelerate launch finetune.py \
    --base_model '/home/wangran108/baichuan' \
    --data_path '/home/wangran108/alpaca-lora-main/all_line_sample_cleaned2.json' \
    --output_dir './baichuan_lora2' \
    --micro_batch_size 4 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --cutoff_len 2048 \
    --val_set_size 0.01 \
    --lora_r 8 \
    --lora_alpha 16 \
    --logging_steps 1 \
    --lora_dropout 0.05 \
    --lora_target_modules '[W_pack]' \
> log.log 2>&1 &
    # --batch_size 32 \

# --train_on_inputs \
# 
# W_pack
# [q_proj,v_proj]
#/home/wangran108/code/llama-7b-hf
#/home/wangran108/code/BELLE-LLAMA-7B-2M
# dialog_instruction
#/home/wangran108/code/belle_13b
# --train_on_inputs 
# bs 8 mbs 1 lr 3e-2
    # --data_path '/home/wangran108/code/alpaca-lora-main/alpaca_gpt4_data_zh.json' \
    # --data_path '/home/wangran108/code/alpaca-lora-main/alpaca_gpt4_data_zh.json' \

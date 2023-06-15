export CUDA_VISIBLE_DEVICES=0
RootDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

Dataset=LDC2020
# Dataset=LDC2017

BasePath=/mlx_devbox/users/gbf/playground/mixed_decoder_KD_loss_exponential_decay        # change dir here
DataPath=/mlx_devbox/users/gbf/playground/AMR-Process_gp2022/outputs/$Dataset

ModelCate=AMRBART-large

MODEL=xfbai/AMRBART-large-v2
ModelCache=$BasePath/.cache
DataCache=$DataPath/.cache/dump-amrparsing

lr=2e-5

OutputDir=${RootDir}/outputs/$Dataset-${ModelCate}-AMRParing-bsz128-lr-${lr}-UnifiedInp-gate-add-model_gen_sample0.3_exponential_scheduled_v2_bfs_NLR

if [ ! -d ${OutputDir} ];then
  mkdir -p ${OutputDir}
else
  read -p "${OutputDir} already exists, delete origin one [y/n]?" yn
  case $yn in
    [Yy]* ) rm -rf ${OutputDir}; mkdir -p ${OutputDir};;
    [Nn]* ) echo "exiting..."; exit;;
    * ) echo "Please answer yes or no.";;
  esac
fi

export HF_DATASETS_CACHE=$DataCache

if [ ! -d ${DataCache} ];then
  mkdir -p ${DataCache}
fi

# torchrun --nnodes=1 --nproc_per_node=1 --max_restarts=0 --rdzv_id=1 --rdzv_backend=c10d main.py \
python3 -u main.py \
    --data_dir $DataPath \
    --task "text2amr" \
    --train_file $DataPath/train.jsonl \
    --validation_file $DataPath/val.jsonl \
    --test_file $DataPath/test.jsonl \
    --output_dir $OutputDir \
    --cache_dir $ModelCache \
    --data_cache_dir $DataCache \
    --tokenizer_name "facebook/bart-large" \
    --model_name_or_path $MODEL \
    --overwrite_output_dir \
    --overwrite_cache \
    --unified_input True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 24 \
    --gradient_accumulation_steps 8 \
    --learning_rate $lr \
    --optim "adamw_hf" \
    --lr_scheduler_type "polynomial" \
    --warmup_steps 200 \
    --num_train_epochs 30 \
    --early_stopping 10 \
    --max_source_length 400 \
    --max_target_length 1024 \
    --val_max_target_length 1024 \
    --generation_max_length 1024 \
    --generation_num_beams 5 \
    --label_smoothing_factor 0.1 \
    --evaluation_strategy "epoch" \
    --weight_decay 0.01 \
    --max_grad_norm 0 \
    --max_steps -1 \
    --predict_with_generate \
    --smart_init False \
    --use_fast_tokenizer False \
    --logging_dir $OutputDir/logs \
    --logging_first_step True \
    --logging_steps 20 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --seed 42 \
    --fp16 \
    --fp16_backend "auto" \
    --dataloader_num_workers 8 \
    --eval_dataloader_num_workers 2 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_smatch" \
    --include_inputs_for_metrics \
    --greater_is_better True \
    --do_train \
    --do_eval \
    --do_predict \
    --ddp_find_unused_parameters False \
    --report_to "tensorboard" \
    --dataloader_pin_memory True 2>&1 | tee $OutputDir/run.log

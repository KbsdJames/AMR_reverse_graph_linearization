gold_amr=/mlx_devbox/users/gbf/playground/AMR-Process_gp2022/outputs/LDC2017/val-gold.amr
hyp_amr=/mlx_devbox/users/gbf/playground/mixed_decoder_KD_loss/fine-tune/outputs/LDC2017-AMRBART-large-AMRParing-bsz64-lr-1e-5-UnifiedInp-gate-add-model_gen_nosmart_init/val_outputs/val_generated_predictions_17115.txt

python3 eval_smatch.py $gold_amr $hyp_amr
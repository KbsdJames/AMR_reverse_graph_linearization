gold_amr=data/to/val-gold.amr
hyp_amr=data/to/generated_amr_file

python3 eval_smatch.py $gold_amr $hyp_amr
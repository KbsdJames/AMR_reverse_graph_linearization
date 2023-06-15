import pdb
from pathlib import Path
import smatch


def calculate_smatch(test_path, predictions_path) -> dict:
    with Path(predictions_path).open() as p, Path(test_path).open() as g:
        score = next(smatch.score_amr_pairs(p, g))
    return {"smatch": score[2]}
res = calculate_smatch('/mlx_devbox/users/gbf/playground/amr_pretraining_encoder/fine-tune/outputs/Eval-LDC2017-AMRBART-large-AMRParing-bsz16-lr-1e-5-UnifiedInp/val_outputs/test_generated_predictions_0.txt', '/mlx_devbox/users/gbf/playground/AMR-Process_gp2022/outputs_debug/LDC2017/train-gold.amr')
print(res["smatch"])
# 0.8845568335588634
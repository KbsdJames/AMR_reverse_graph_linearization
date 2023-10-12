# RGL-AMR
The implementation for Findings of EMNLP2023 paper "Guiding AMR Parsing with Reverse Graph Linearization". [Paper Link]() (Arxiv). 

We thank authors of [AMRBART](https://github.com/goodbai-nlp/AMRBART/tree/acl2022) for releasing their code. Our implementation is based on their repository.


# Data

You may download the AMR corpora at [LDC](https://www.ldc.upenn.edu).

Please follow [this respository](https://github.com/goodbai-nlp/AMR-Process) to preprocess AMR graphs. For the reverse linearized data, we will release the code soon.

> We release some examples for preprocessed data at AMR_reverse_graph_linearization/data. If you have the license, feel free to contact us for getting the preprocessed data.


# Train an AMR parser with RGL

After **After configuring the path of the scripts**, run
```
cd fine-tune
bash train_rgl.sh
```




# Evaluation
```
cd fine-tune/evaluation
bash eval_smatch.sh
```
For better results, you can postprocess the predicted AMRs using the [BLINK](https://github.com/facebookresearch/BLINK) tool following [SPRING](https://github.com/SapienzaNLP/spring).

# ULTRA-DP: Unifying Graph Pre-training with Multi-task Graph Dual Prompt


We provide the code (in pytorch) for our work: "ULTRA-DP: Unifying Graph Pre-training with Multi-task Graph Dual Prompt".

```bash
pip install -r requirements.txt
```

## Pre-process Datasets

```bash
python preprocess_dataset.py --dataset DBLP
```

After that, a graph data will be saved into `datadrive/dataset/graph_dblp.pk`. Valid values of `dataset` include `['DBLP', 'Pubmed', 'CoraFull', 'Coauthor-CS']`. 

## Pre-process Reachability Matrices

```bash
python preprocess_reachability.py \
    --data_dir datadrive/dataset/graph_dblp.pk
```

## Training

- Pre-train + Fine-tune with ULTRA-DP (Edge + k-NN) with a prompt node for 300 epochs:

```bash
python train.py \
    --data_dir datadrive/dataset/graph_dblp.pk \
    --pre_training_task hybrid-knn_6-link \
    --pretrain_model_dir datadrive/model/ultra_dp_dblp \
    --prompt_size 1 \
    --position_anchor_num 0.01 \
    --pretrain_epoch 300 \
    --few_shot 32 \
```

Final pre-trained weights will be saved into `datadrive/model/ultra_dp_dblp`.

- Pre-train + Fine-tune with ULTRA-DP (Edge + k-NN) without prompt:

Set `prompt_size` as 0 (default value)

```bash
python train.py \
    --data_dir datadrive/dataset/graph_dblp.pk \
    --pre_training_task hybrid-knn_6-link \
    --pretrain_model_dir datadrive/model/hybrid_dp_dblp \
    --pretrain_epoch 300 \
    --few_shot 32
```

- Using pre-trained weights to fine-tune GNNs:

Set `pretrain_model_dir` as the weight path and `pretrain_epoch` as 0.

```bash
python train.py \
    --data_dir datadrive/dataset/graph_dblp.pk \
    --pre_training_task hybrid-knn_6-link \
    --pretrain_model_dir datadrive/model/ultra_dp_dblp \
    --prompt_size 1 \
    --position_anchor_num 0.01 \
    --few_shot 32 \
    --pretrain_epoch 0
```

- Directly fine-tune GNNs without pre-training

Set `pretrain_model_dir` as a non-exist file and `pretrain_epoch` as 0.

```bash
python train.py \
    --data_dir datadrive/dataset/graph_dblp.pk \
    --pretrain_model_dir datadrive/model/temp_model \
    --few_shot 32 \
    --pretrain_epoch 0
```

More hyper-parameters can be found in `train.py`.

## Citation

```
@article{chen2023ultra,
  title={ULTRA-DP: Unifying Graph Pre-training with Multi-task Graph Dual Prompt},
  author={Chen, Mouxiang and Liu, Zemin and Liu, Chenghao and Li, Jundong and Mao, Qiheng and Sun, Jianling},
  journal={arXiv preprint arXiv:2310.14845},
  year={2023}
}
```


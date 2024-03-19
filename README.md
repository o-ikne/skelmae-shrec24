# **Shrec24**
[SHREC24] Skeleton-based Self-Supervised Learning For Dynamic Hand Gesture Recognition

![hippo](images/mae_approach.jpg)

## **Updates**
- code will be available soon

## **Installation**
Create and activate conda environment:
```
conda create -n shrec24 python=3.10
conda activate shrec24
```

Install all dependencies:
```
pip install -r requirements.txt
```

## Training
Download the [**SHREC'24**](https://www.shrec.net/SHREC-2024-hand-motion/) dataset.

### Training

```
bash train.sh --config_file configs/shrec24_config.yaml
```

## Evaluation

```
bash eval.sh --config_file configs/shrec24_config.yaml
```

We thank [MAE](https://github.com/facebookresearch/mae) and [STGCN](https://github.com/yysijie/st-gcn) for making their code available

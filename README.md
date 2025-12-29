
# **HEAR**: Hierarchically Enhanced Aesthetic Representations for Multidimensional Music Evaluation
Official PyTorch Implementation of ICASSP 2026 paper "HEAR: Hierarchically Enhanced Aesthetic Representations for Multidimensional Music Evaluation"

This repository contains the training and evaluation code for HEAR, a robust framework designed to address the challenges of multidimensional music aesthetic evaluation under limited labeled data.
![](figs/HEAR.png)
## 🌟 Key Features
* **SOTA Performance**: Ranked 2nd/19 on Track 1 and 5th/17 on Track 2 in the [ICASSP 2026 Automatic Song Aesthetics Evaluation Challenge](https://aslp-lab.github.io/Automatic-Song-Aesthetics-Evaluation-Challenge/).
* **Robustness**: Synergizes Multi-Source Multi-Scale Representations and Hierarchical Augmentation to capture robust features under limited labeled data.
* **Dual Capability**: Optimized for both exact score prediction and ranking (Top-Tier Identification).

## 🚀 Quick Start
TODO: Package as a command-line tool

## 📦 Installation
Clone the repository and install dependencies:
```
git clone https://github.com:Eps-Acoustic-Revolution-Lab/EAR_HEAR.git
git submodule update --init --recursive

conda create -n hear python=3.10 -y
conda activate hear
pip install -r requirements.txt
```

## 🎯 Training

### Step 1: Data Preparation

First, prepare the dataset by running the data pipeline:

```bash
cd data_pipeline
bash run.sh
```

This script will:
1. **Download Dataset**: Download the [SongEval](https://huggingface.co/datasets/ASLP-lab/SongEval) dataset
2. **Split Dataset**: Split the dataset into training and validation sets based on [the challenge's validation IDs
](https://github.com/ASLP-lab/Automatic-Song-Aesthetics-Evaluation-Challenge/blob/main/static/val_ids.txt)
3. **Audio Augmentation**: Apply audio augmentation to the training set
4. **Extract Features**: Extract MuQ and MusicFM features for both training and test sets
5. **Generate PKL Files**: Generate `train_set.pkl` and `test_set.pkl` files for training and evaluation


### Step 2:




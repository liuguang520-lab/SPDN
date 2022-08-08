
## Usage

### Prerequisites
- Python >= 3.5
- Pytorch >= 1.3 and torchvision (https://pytorch.org/)
- You can use the `requirements.txt` file we provide to setup the environment via Anaconda.
```
conda create --name py36 python=3.6
conda install pytorch torchvision -c pytorch
pip3 install -r requirements.txt
```

### Install
Clone this repository:
```
git clone https://github.com/hytseng0509/CrossDomainFewShot.git
cd CrossDomainFewShot
```

### Datasets
Download 5 datasets seperately with the following commands.
- Set `DATASET_NAME` to: `cars`, `cub`, `miniImagenet`, `places`, or `plantae`.
```
cd filelists
python3 process.py DATASET_NAME
cd ..
```
- Refer to the instruction [here](https://github.com/wyharveychen/CloserLookFewShot#self-defined-setting) for constructing your own dataset.

### Feature encoder pre-training
We adopt `baseline++` for MatchingNet, and `baseline` from [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot) for other metric-based frameworks.
- Download the pre-trained feature encoders.
```
cd output/checkpoints
python3 download_encoder.py
cd ../..
```
- Or train your own pre-trained feature encoder (specify `PRETRAIN` to `baseline++` or `baseline`).
```
python3 train_baseline.py --method PRETRAIN --dataset miniImagenet --name PRETRAIN --train_aug
```

### Training with multiple seen domains
Baseline training w/o feature-wise transformations.
- `METHOD` : metric-based framework `gnnnet`.
- `TESTSET`: unseen domain `cars`, `cub`, `places`, or `plantae`.
```
python3 train_baselineSSL.py --method METHOD --dataset multi --testset TESTSET --name multi_TESTSET_ori_METHOD --warmup PRETRAIN --train_aug
```



### Evaluation
Test the metric-based framework `METHOD` on the unseen domain `TESTSET`.
- Specify the saved model you want to evaluate with `--name` (e.g., `--name multi_TESTSET_lft_METHOD` from the above example).
```
python3 test.py --method METHOD --name NAME --dataset TESTSET
```


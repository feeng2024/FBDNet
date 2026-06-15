# Disentangling Foreground–Background Representations for Robust Camouflaged Object Detection

Official PyTorch implementation of:

> **Disentangling Foreground–Background Representations for Robust Camouflaged Object Detection**

This repository provides the complete code for network construction, training, testing, evaluation, and visualization.

## Requirements

The code was tested with:

```text
Python 3.x
PyTorch 2.0.1
Torchvision 0.15.2
Timm 1.0.15
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Datasets

The experiments use the following datasets:

* COD: COD10K, CAMO, and NC4K
* SOD: DUTS-TE, DUT-OMRON, ECSSD, HKU-IS, and PASCAL-S

Organize the training data as:

```text
Dataset/
└── TrainDataset/
    ├── Imgs/
    └── GT/
```

Organize each testing dataset as:

```text
Dataset/
└── TestDataset/
    └── COD10K/
        ├── Imgs/
        └── GT/
```

You can find these datasets [here](https://github.com/lartpang/awesome-segmentation-saliency-dataset#camouflaged-object-detection-cod).


## Pretrained Backbone

We use **P2T-Large** as the backbone network.

Please download the ImageNet-pretrained P2T-Large weights, place them in the `Checkpoints/` directory, and configure the weight path in:

```text
Net/p2t.py
```

The trained model checkpoint can be downloaded from:

> [Model download link]


## Training

Configure the dataset and checkpoint paths in `MyTrain.py`, and then run:

```bash
python MyTrain.py
```

The default backbone is P2T-Large. A full-model checkpoint can also be loaded using:

```bash
python MyTrain.py 
```

## Testing

Configure the model and dataset paths in `MyTest.py`, and run:

```bash
python MyTest.py
```

## Evaluation

Configure the prediction and ground-truth paths in `MyEval.py`, and run:

```bash
python MyEval.py
```

## Visualization Results

Representative prediction maps and visualization results are available at:

> [https://pan.baidu.com/s/1qYVzoyf_RAstOYJyVSYKIg?pwd=2sfi 提取码: 2sfi]


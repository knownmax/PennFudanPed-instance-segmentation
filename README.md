# Mask R-CNN Instance Segmentation — Penn-Fudan Pedestrian Dataset

Fine-tuning a pretrained **Mask R-CNN** (ResNet-50 FPN backbone) on the [Penn-Fudan Pedestrian Dataset](https://www.cis.upenn.edu/~jshi/ped_html/) for pedestrian instance segmentation. Follows the official [PyTorch detection tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

## Task

Given a street-scene image, predict per-pedestrian **bounding boxes**, **class labels**, and **pixel-level instance masks**.

- 2 classes: background (0) and person (1)
- Evaluated with COCO-style AP metrics (bbox + segm)

## Dataset

**Penn-Fudan Pedestrian Database** — 170 images of urban pedestrians with corresponding instance segmentation masks and bounding-box annotations.

```
data/PennFudanPed/
├── PNGImages/       # 170 RGB images
├── PedMasks/        # instance masks (pixel value = pedestrian ID)
└── Annotation/      # per-image bounding-box annotation files
```

Download and unzip manually (or uncomment the wget cell in the notebook):

```bash
wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip -P data
cd data && unzip PennFudanPed.zip
```

## Project Structure

```
.
├── run.ipynb          # main training & inference notebook
├── engine.py          # train_one_epoch / evaluate loop
├── coco_eval.py       # COCO AP evaluator wrapper
├── coco_utils.py      # dataset → COCO API conversion helpers
├── transforms.py      # augmentation transforms (random flip, etc.)
├── utils.py           # metric logger, collate_fn, distributed helpers
└── data/              # dataset (gitignored)
```

## Model

| Component | Details |
|-----------|---------|
| Backbone | ResNet-50 + FPN, pretrained on COCO |
| Head | Mask R-CNN with custom 2-class predictor |
| Optimizer | SGD (lr=0.005, momentum=0.9, weight_decay=0.0005) |
| LR schedule | Step decay ×0.1 every 3 epochs |
| Epochs | 10 |
| Batch size | 2 |

A Faster R-CNN (detection only, no masks) variant and a MobileNetV2-backbone option are also shown in the notebook for comparison.

## Requirements

```
torch
torchvision
pycocotools
```

Install with:

```bash
pip install torch torchvision pycocotools
```

## Usage

Open and run `run.ipynb` cell by cell. Training and evaluation happen in the final cells; intermediate cells visualize images/masks and test forward passes.

## Results

Evaluated on a held-out split (50 images). COCO AP metrics are printed after each epoch by the `evaluate()` function.

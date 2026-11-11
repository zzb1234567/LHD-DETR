# LHD-DETR

Implementation accompanying **LHD-DETR: Linearised hybrid attention for real-time joint detection of tomato fruits and stems**.

**Authors** — Zhibo Zhang, Yuzhou Zhang, Nan Li, Rui Liu, Yifan Zhang, and Zhonghua Miao.

LHD-DETR builds on Deformable DETR for joint detection of tomato fruits and stems in greenhouse images. The Linearised Hybrid Feature Module (LHFM) combines convolution with Linearised Angular Self-Attention (LASA) in the fourth backbone stage to capture local detail and contextual information.

## Results reported in the manuscript

Detection metrics are percentages on the tomato fruit–stem validation set.

| Model | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: |
| Deformable DETR | 79.58 | 45.41 | 45.69 |
| LHD-DETR | **80.38** | **46.27** | **46.91** |

| Target | AP@0.5 | AP@0.5:0.95 |
| --- | ---: | ---: |
| Stems | 69.25 | 27.25 |
| Fruits | 91.51 | 66.57 |

The manuscript reports **13.5 FPS** for the complete detector with 640 × 480 inputs, including preprocessing, inference, and postprocessing. Backbone runtime experiments use separate settings, with 640 × 480 inputs on the CPU and 4032 × 3024 inputs on the GPU.

## Environment

The manuscript reports the following experimental environment.

| Component | Configuration |
| --- | --- |
| Python | 3.8 |
| PyTorch | 2.0.0 |
| CUDA | 11.8 |
| GPU | NVIDIA GeForce RTX 4090 |
| CPU | Intel Xeon Platinum 8352V, 16 vCPUs |

The commands below use a Bash shell and a CUDA-capable environment. Building the custom attention operator requires the CUDA Toolkit, including `nvcc`, and a compatible C++ compiler.

### Install dependencies

Run these commands from the repository root after downloading or cloning the project.

```bash
conda create -n lhd-detr python=3.8 -y
conda activate lhd-detr

python -m pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install "numpy<2" scipy pillow pycocotools tqdm einops
```

The torchvision version follows the official wheel pairing for PyTorch 2.0.0. Auxiliary package versions are installation guidance; the manuscript specifies Python, PyTorch, and CUDA versions. [PyTorch installation reference](https://pytorch.org/get-started/previous-versions/)

### Build multi-scale deformable attention

Ensure `nvcc --version` reports the intended CUDA Toolkit and PyTorch can access the GPU.

```bash
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

cd models/ops
bash make.sh
python test.py
cd ../..
```

The tests compare the custom operator with its PyTorch implementation and check numerical gradients. The build script requires CUDA to be available.

## Dataset

The manuscript uses a tomato fruit–stem dataset with two categories.

| Split | Images | Fruit instances | Stem instances |
| --- | ---: | ---: | ---: |
| Training | 1,995 | 10,782 | 4,781 |
| Validation | 221 | 1,096 | 536 |
| Total | 2,216 | 11,878 | 5,317 |

The dataset is available from the corresponding author upon reasonable request, as stated in the manuscript.

Prepare COCO-format bounding-box annotations with the following directory structure.

```text
tomato_dataset/
├── train2017/
│   └── training images
├── val2017/
│   └── validation images
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

Set `--coco_path` to the `tomato_dataset` directory. Each annotation's `image_id` must reference an image entry, whose `file_name` resolves inside the corresponding image directory. Bounding boxes use COCO's `[x, y, width, height]` convention.

**Category IDs must be 0 and 1.** The current model has two output classes and reads annotation category IDs directly. Assign fruit and stem consistently in both splits and in the JSON `categories` entries. If the source annotations use IDs 1 and 2, remap the IDs in both `annotations` and `categories` before training.

## Training and validation

The following single-GPU command uses the training settings reported in the manuscript. Replace `/path/to/tomato_dataset` with your dataset directory.

```bash
python main.py \
    --dataset_file coco \
    --coco_path /path/to/tomato_dataset \
    --output_dir outputs/lhd_detr \
    --device cuda \
    --batch_size 4 \
    --epochs 100 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --lr_drop 70 \
    --set_cost_class 2 \
    --set_cost_bbox 5 \
    --set_cost_giou 2 \
    --cls_loss_coef 2 \
    --bbox_loss_coef 5 \
    --giou_loss_coef 2
```

The default optimizer is AdamW. The default prediction configuration is single-stage, with iterative bounding-box refinement disabled. The backbone uses the 50-layer configuration, and LHFM uses an angular mapping exponent of 3.

At construction, the backbone loads matching ImageNet-pretrained ResNet-50 parameters through torchvision. The first run may download these weights. With the default trainable-backbone configuration, the stem and `layer1` are frozen, while `layer2`, `layer3`, and `layer4` are trainable.

Training augmentation combines horizontal flipping, resizing, and cropping. Validation resizes the shorter image side to 800 pixels while limiting the longer side to 1333 pixels.

Validation runs automatically after each epoch using the model currently in memory. With an output directory supplied, the training script writes the following files.

| File | Contents |
| --- | --- |
| `checkpoint.pth` | Latest model, optimizer, scheduler, epoch, and arguments |
| `checkpoint_best.pth` | Best model checkpoint selected by validation mAP@0.5:0.95 |
| `log.txt` | Per-epoch training and validation metrics in JSON-lines format |
| `eval/` | Saved COCO evaluation outputs |

In `log.txt`, the first three entries of `test_coco_eval_bbox` are AP@0.5:0.95, AP@0.5, and AP@0.75. Multiply these values by 100 to express them as percentages.

### Checkpoint loading in the current implementation

The continuous training workflow above performs validation directly on the trained model. The current `main.py --resume` path filters out `layer4` checkpoint entries, including when `--eval` is supplied. It also leaves optimizer, scheduler, and epoch restoration disabled. Standalone evaluation of a saved model requires updating that loading path to restore the complete trained model.

## Code overview

| Path | Purpose |
| --- | --- |
| `main.py` | Training setup, checkpoint handling, and validation |
| `engine.py` | Training and evaluation loops |
| `models/linearised_hybrid.py` | LHFM implementation |
| `models/ResNet_ImageNet.py` | Hybrid backbone definition |
| `models/backbone.py` | Backbone initialization and feature extraction |
| `models/deformable_detr.py` | Detector, losses, and output processing |
| `models/ops/` | Custom multi-scale deformable attention operator |
| `datasets/` | COCO-format loading, augmentation, and evaluation |
| `tools/` | Distributed launch utilities |

## Acknowledgements

This implementation builds on Deformable DETR and DETR. Retain the upstream copyright and license notices when redistributing derived code. [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) · [DETR](https://github.com/facebookresearch/detr)

## Contact

For dataset requests and questions about the study, contact the corresponding author, Zhonghua Miao, at `zhonghuamiao@163.com`.

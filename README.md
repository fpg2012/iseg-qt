# iseg-qt

an interactive segmentation demo app written in PySide6

## Basic Usage

### For SAM

First, install [segment-anything](https://github.com/facebookresearch/segment-anything) and prepare the model checkpoint.

In the directory of SAM repository:

```
git clone https://github.com/fpg2012/iseg-qt --depth 1
cd iseg_qt
python iseg_sam.py --device cpu --checkpoint <path to the ViT-H checkpoint>
```

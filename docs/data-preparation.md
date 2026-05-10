# Data Preparation

All datasets must be converted to Apache Arrow format before training. Conversion scripts live in `renaissance/utils/write_*.py`. Run the `make_arrow(root, arrows_root)` function for each dataset you need and place the outputs under `data/arrow/`.

> **Note:** We do not distribute datasets. Download them from the official sources listed below.

---

## Directory layout

After conversion, `data/arrow/` should contain one subdirectory per dataset:

```
data/arrow/
├── coco_caption_karpathy_train.arrow
├── coco_caption_karpathy_val.arrow
├── coco_caption_karpathy_test.arrow
├── vg_caption_train.arrow
├── vg_caption_val.arrow
├── nlvr2_train.arrow
├── nlvr2_dev.arrow
├── nlvr2_test1.arrow
├── vqav2_train.arrow
├── vqav2_val.arrow
├── vqav2_test.arrow
└── ...
```

Each `.arrow` file is a serialised `datasets.Dataset` (HuggingFace Datasets format).

---

## Datasets

### COCO Captions

Download: [2014 train images](http://images.cocodataset.org/zips/train2014.zip), [2014 val images](http://images.cocodataset.org/zips/val2014.zip), and [Karpathy split](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).

```
data/coco/
├── train2014/
│   └── COCO_train2014_*.jpg
├── val2014/
│   └── COCO_val2014_*.jpg
└── karpathy/
    └── dataset_coco.json
```

```python
from renaissance.utils.write_coco_karpathy import make_arrow
make_arrow("data/coco/", "data/arrow/")
```

---

### Visual Genome (VG)

Download: [image part 1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [image part 2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip), and [region descriptions](http://visualgenome.org/static/data/dataset/region_descriptions.json.zip).

```
data/vg/
├── images/
│   ├── VG_100K/
│   └── VG_100K_2/
└── annotations/
    └── region_descriptions.json
```

```python
from renaissance.utils.write_vg import make_arrow
make_arrow("data/vg/", "data/arrow/")
```

---

### Flickr30K

Sign the [request form](https://forms.illinois.edu/sec/229675) and download the [Karpathy split](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).

```
data/flickr30k/
├── flickr30k-images/
│   └── *.jpg
└── karpathy/
    └── dataset_flickr30k.json
```

```python
from renaissance.utils.write_f30k_karpathy import make_arrow
make_arrow("data/flickr30k/", "data/arrow/")
```

---

### VQAv2

Download COCO 2014 train/val images (see COCO above), [2015 test images](http://images.cocodataset.org/zips/test2015.zip), annotations ([train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip)), and questions ([train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip), [test](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip)).

```
data/vqa/
├── train2014/ (symlink or copy from COCO)
├── val2014/
├── test2015/
├── v2_OpenEnded_mscoco_train2014_questions.json
├── v2_OpenEnded_mscoco_val2014_questions.json
├── v2_OpenEnded_mscoco_test2015_questions.json
├── v2_mscoco_train2014_annotations.json
└── v2_mscoco_val2014_annotations.json
```

```python
from renaissance.utils.write_vqa import make_arrow
make_arrow("data/vqa/", "data/arrow/")
```

---

### NLVR2

Clone the [NLVR2 repository](https://github.com/lil-lab/nlvr) and sign the [image request form](https://goo.gl/forms/yS29stWnFWzrDBFH3).

```
data/nlvr2/
├── images/train/
│   ├── 0/
│   └── ...
├── dev/
├── test1/
├── nlvr/
└── nlvr2/
```

```python
from renaissance.utils.write_nlvr2 import make_arrow
make_arrow("data/nlvr2/", "data/arrow/")
```

---

### SNLI-VE

Built on top of Flickr30K images (see above) and [SNLI](https://nlp.stanford.edu/projects/snli/).

```python
from renaissance.utils.write_snli import make_arrow
make_arrow("data/snli/", "data/arrow/")
```

---

### Google Conceptual Captions (GCC)

Download from [https://ai.google.com/research/ConceptualCaptions/download](https://ai.google.com/research/ConceptualCaptions/download). Note that many image URLs are no longer accessible — write your own downloader.

```
data/gcc/
├── images_train/
├── images_val/
├── train_annot.json
└── val_annot.json
```

```python
from renaissance.utils.write_conceptual_caption import make_arrow
make_arrow("data/gcc/", "data/arrow/")
```

---

### SBU Captions

Download from [http://www.cs.virginia.edu/~vicente/sbucaptions/](http://www.cs.virginia.edu/~vicente/sbucaptions/).

```python
from renaissance.utils.write_sbu import make_arrow
make_arrow("data/sbu/", "data/arrow/")
```

---

## Configuring datasets for training

Set `data.datasets` in your YAML config or via CLI:

```yaml
data:
  datasets:
    - coco
    - vg
  data_root: data/arrow/
```

```bash
python run.py configs/pretrain_two_tower.yaml data.datasets=[coco,vg] data.data_root=data/arrow/
```

Available dataset keys: `coco`, `vg`, `gcc`, `sbu`, `f30k`, `vqa`, `nlvr2`, `snli`.

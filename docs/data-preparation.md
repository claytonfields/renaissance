# Data Preparation

> **Note:** We do not distribute datasets. Download them from the official sources listed below.

## Data backends

Renaissance has two data backends, selected by `data.backend`:

- **`modern`** *(default since the 1.3 line)* — `renaissance/data`, HuggingFace
  Hub-first. Datasets are streamed/loaded directly via `datasets.load_dataset`;
  **no Arrow pre-conversion step is required**. This is the recommended path
  for new work. See [Configuring datasets for training](#configuring-datasets-for-training).
- **`legacy`** *(deprecated)* — `renaissance/datamodules` + `renaissance/datasets`,
  backed by pre-serialized Apache Arrow files and PyTorch Lightning. Kept only
  so existing on-disk Arrow data keeps working until the legacy layer is
  removed. Importing it emits a `DeprecationWarning`.

### Migrating off the legacy backend

If you have existing `data/arrow/` files (produced by the old
`make_arrow` scripts) you have two options:

1. **Keep using them for now** — pin `data.backend: legacy` in your config
   (the shipped `configs/*.yaml` already do this) and your existing
   `data_root: data/arrow/` continues to work unchanged.
2. **Switch to `modern`** — drop the Arrow step entirely; set
   `data.backend: modern` (or omit it — it is the default) and let the
   modern loaders pull each dataset from the Hub. The dataset keys are the
   same (`coco`, `vqa`, `nlvr2`, …); per-dataset Hub options go in
   `data.dataset_kwargs`.

The Arrow conversion scripts below apply **only to the legacy backend** and
now live under `renaissance/utils/legacy/write_*.py`. Run the
`make_arrow(root, arrows_root)` function for each dataset and place the
outputs under `data/arrow/`.

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
# modern backend (default) — no Arrow step, loaded from the Hub
data:
  datasets:
    - coco
    - vg

# legacy backend — reads pre-serialized Arrow under data_root
data:
  backend: legacy
  datasets:
    - coco
    - vg
  data_root: data/arrow/
```

```bash
python run.py configs/pretrain_two_tower.yaml data.datasets=[coco,vg]
```

Available dataset keys: `coco`, `vg`, `gcc`, `sbu`, `f30k`, `vqa`, `nlvr2`, `snli` (both backends), plus modern-only `coco_karpathy`, `cc3m`, `cc12m`, `refcoco`, `refcocoplus`, `refcocog`, `glue`.

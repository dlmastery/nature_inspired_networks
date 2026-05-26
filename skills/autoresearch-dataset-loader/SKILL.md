---
name: autoresearch-dataset-loader
description: Use when wiring up a new benchmark dataset (image / tabular / graph) into an autoresearch project. Covers torchvision / HuggingFace / MedMNIST / WILDS / OGB patterns, plus the Python 3.13 SSL workaround for corporate networks.
---

# Skill — Wire up a benchmark dataset

## When to use

Any time you need to add a new dataset to a sweep. Adding one
correctly takes ~30 min; getting it wrong (e.g., train/test leakage,
mis-normalised pixels, wrong label index) corrupts every downstream
metric.

## Contract for `load_dataset(name, root, batch_size, num_workers)`

Returns the 4-tuple `(train_loader, test_loader, num_classes,
in_channels)`. Every dataset wrapper conforms to this contract so
the runner is dataset-agnostic.

## Image-classification template

```python
from torch.utils.data import DataLoader
import torchvision.transforms as T

DATASET_MEAN = (0.4914, 0.4822, 0.4465)
DATASET_STD  = (0.2470, 0.2435, 0.2616)

def _tfs(train: bool):
    if train:
        return T.Compose([
            T.RandomCrop(IMG_SIZE, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(DATASET_MEAN, DATASET_STD),
        ])
    return T.Compose([T.ToTensor(),
                      T.Normalize(DATASET_MEAN, DATASET_STD)])

def load_dataset(root, batch_size, num_workers):
    tr = DSClass(root=root, train=True,  download=True,
                 transform=_tfs(True))
    te = DSClass(root=root, train=False, download=True,
                 transform=_tfs(False))
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True,
                   drop_last=True,
                   persistent_workers=num_workers > 0),
        DataLoader(te, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True,
                   persistent_workers=num_workers > 0),
        NUM_CLASSES,
        IN_CHANNELS,
    )
```

## Python 3.13 SSL workaround (corp networks)

Python 3.13 ships a stricter SSL implementation that rejects corporate
CA chains with `Basic Constraints of CA cert not marked critical`.
Bypass:

```powershell
# 1. Download the tarball out-of-band with curl
curl.exe -kL -o data/<dataset>.tar.gz https://<host>/<path>

# 2. Let the dataset wrapper verify md5 and extract
python -c "from torchvision.datasets import <DS>; \
  <DS>(root='./data', train=True, download=True)"
```

The `download=True` step is now a no-op because the tarball already
exists with the right MD5; torchvision extracts and proceeds.

## Windows DataLoader workaround

`num_workers > 0` on Windows uses the spawn start-method. Workers
re-import the parent module, which can wedge silently if any pickled
transform has a closure variable or if a CUDA context was created
before fork.

**Default to `num_workers: 0` on Windows.** Re-enable per-config on
Linux / WSL only.

## MedMNIST template

```python
def medmnist_loaders(root, flag, batch_size, num_workers, size=28):
    import medmnist
    from medmnist import INFO
    info = INFO[flag]
    DataClass = getattr(medmnist, info["python_class"])
    n_cls = len(info["label"])
    tfs = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        T.Normalize((0.5,)*3, (0.5,)*3),
    ])
    tr = DataClass(split="train", transform=tfs, download=True,
                   root=root, size=size)
    te = DataClass(split="test",  transform=tfs, download=True,
                   root=root, size=size)
    return DataLoader(tr, batch_size, shuffle=True), \
           DataLoader(te, batch_size, shuffle=False), n_cls, 3
```

Note: `T.Lambda` with a `lambda` is unpicklable; if you raise
`num_workers > 0` on Linux you must replace it with a named function.

## Mandatory checks before declaring "done"

1. `len(train_loader.dataset) == <documented size>` — sanity vs.
   dataset card.
2. `len(test_loader.dataset)` matches the documented size.
3. First batch: `x.shape == (B, C, H, W)`, `x.dtype == torch.float32`,
   `0 < x.mean() < 1` if normalisation is applied.
4. Train and test sets have *no* shared sample IDs (run a hash
   check on raw bytes for ~100 random samples).
5. Per-class counts on train: print and visually verify no class is
   massively imbalanced.

## Anti-patterns

- Hard-coding `data/` paths inside the dataset wrapper. Accept
  `root` and return the loader.
- Forgetting `pin_memory=True` on GPU systems — costs ~10% throughput.
- Setting `drop_last=False` on train loader — causes BatchNorm
  variance issues on the partial last batch.
- Mixing train normalisation stats with test stats. Compute stats
  on train, apply to both.

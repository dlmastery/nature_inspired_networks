"""Dataset loaders: CIFAR-10/100 + MedMNIST 2D + Imagenette wrappers.

All loaders return (train_loader, test_loader, num_classes, in_channels).
Mean/std normalization values are dataset-standard.
"""
from __future__ import annotations

import os
import ssl
import sys
import tarfile
import urllib.request
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

# ImageNet normalisation stats — used by Imagenette (a 10-class
# subset of ImageNet). These are the standard ImageNet mean/std and
# match the timm / torchvision pre-trained-model convention.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Imagenette v2-160 (160px shortest-edge variant) download URL.
# Hosted by fast.ai on S3. ``imagenette2-160.tgz`` is ~98 MB. The
# tarball extracts to a directory called ``imagenette2-160/`` with
# the standard ImageFolder layout (train/<class>/<jpg> + val/<class>/<jpg>).
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_DIRNAME = "imagenette2-160"
IMAGENETTE_TGZ_NAME = "imagenette2-160.tgz"


def _cifar_tfs(mean, std, train: bool,
               randaugment_n: int = 0, randaugment_m: int = 14,
               random_erasing_p: float = 0.0):
    """Build the CIFAR train/test transform pipeline.

    The default ``randaugment_n=0`` / ``random_erasing_p=0.0`` reproduce
    the legacy pipeline byte-for-byte. Setting ``randaugment_n>0`` adds
    a :class:`torchvision.transforms.RandAugment` between the crop/flip
    and ToTensor; setting ``random_erasing_p>0`` appends a
    :class:`torchvision.transforms.RandomErasing` after Normalize.

    These two arguments are the only modern-recipe entry points that
    touch the DataLoader pipeline; Mixup / CutMix / EMA live in the
    Trainer (per-batch / per-step) and need no pipeline change.
    """
    if train:
        steps: list = [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
        ]
        if randaugment_n and randaugment_n > 0:
            from .randaugment import build_randaugment
            steps.append(build_randaugment(num_ops=int(randaugment_n),
                                           magnitude=int(randaugment_m)))
        steps.extend([T.ToTensor(), T.Normalize(mean, std)])
        if random_erasing_p and random_erasing_p > 0.0:
            from .random_erasing import build_random_erasing
            steps.append(build_random_erasing(p=float(random_erasing_p)))
        return T.Compose(steps)
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


def cifar_loaders(root: str = "./data", batch_size: int = 256, num_workers: int = 4,
                  variant: str = "cifar10",
                  randaugment_n: int = 0, randaugment_m: int = 14,
                  random_erasing_p: float = 0.0,
                  headline_mode: bool = False, seed: int = 0):
    root = Path(root)
    if variant == "cifar10":
        Ds = CIFAR10
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        n_cls = 10
    elif variant == "cifar100":
        Ds = CIFAR100
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        n_cls = 100
    else:
        raise ValueError(variant)
    tr = Ds(root=str(root), train=True, download=True,
            transform=_cifar_tfs(mean, std, train=True,
                                 randaugment_n=randaugment_n,
                                 randaugment_m=randaugment_m,
                                 random_erasing_p=random_erasing_p))
    te = Ds(root=str(root), train=False, download=True,
            transform=_cifar_tfs(mean, std, train=False))
    # Synthesis-100 D4 / D5 (2026-06-06): in headline_mode wire a
    # per-loader generator + ``worker_init_fn`` so DataLoader workers
    # use a deterministic RNG seeded by the per-run ``seed``. Without
    # this, RandAugment / RandomErasing produce non-reproducible
    # augmentations even though every other RNG is seeded. ``seed_worker``
    # lives in ``runner`` to avoid a circular import here we resolve it
    # lazily.
    train_kwargs: dict = dict(
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    if headline_mode:
        from .runner import seed_worker
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        train_kwargs["generator"] = gen
        train_kwargs["worker_init_fn"] = seed_worker
    tr_loader = DataLoader(tr, **train_kwargs)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=num_workers > 0)
    return tr_loader, te_loader, n_cls, 3


def medmnist_loaders(root: str = "./data/medmnist", flag: str = "pathmnist",
                     batch_size: int = 256, num_workers: int = 4, size: int = 28):
    """Load a MedMNIST 2D dataset. flag in {pathmnist, organamnist, octmnist, ...}."""
    import medmnist  # type: ignore
    from medmnist import INFO

    info = INFO[flag]
    n_cls = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])
    Path(root).mkdir(parents=True, exist_ok=True)
    mean = (0.5,) * (3 if size > 28 else 3)
    std = (0.5,) * 3
    tfs = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        T.Normalize(mean, std),
    ])
    tr = DataClass(split="train", transform=tfs, download=True, root=root, size=size)
    te = DataClass(split="test", transform=tfs, download=True, root=root, size=size)
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True, drop_last=True)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    return tr_loader, te_loader, n_cls, 3


class _RotatedCIFAR(torch.utils.data.Dataset):
    """Wraps a CIFAR-10/100 dataset with deterministic per-image rotation
    chosen from a fixed set of angles.

    Used by :func:`load_rotated_cifar10` to build the eval pipeline
    (where rotations must be deterministic so the metric is stable
    across runs) and the all-4-rotations TTA pipeline.
    """

    def __init__(self, base, angles: tuple[int, ...] = (0, 90, 180, 270),
                 fill: int = 0):
        self.base = base
        self.angles = tuple(int(a) for a in angles)
        self.fill = float(fill)

    def __len__(self) -> int:
        return len(self.base) * len(self.angles)

    def __getitem__(self, idx: int):
        n = len(self.angles)
        base_idx = idx // n
        a_idx = idx % n
        img, label = self.base[base_idx]
        angle = self.angles[a_idx]
        if angle != 0:
            img = T.functional.rotate(img, angle, fill=[self.fill] * 3)
        return img, label


def rotated_cifar_loaders(root: str = "./data", batch_size: int = 256,
                          num_workers: int = 0, variant: str = "cifar10",
                          rotation_degrees: tuple[int, ...] = (0, 90, 180, 270),
                          fill: int = 0):
    """Load CIFAR-10/100 with rotation augmentation.

    Train pipeline applies a stochastic :class:`torchvision.transforms.
    RandomRotation` over ``rotation_degrees`` (treated as a discrete set
    rather than a continuous range -- the union of the four cardinal
    rotations is the canonical rotation-equivariance benchmark for
    H71's CIFAR proxy).

    Eval pipeline applies ALL four rotations as test-time augmentation
    via :class:`_RotatedCIFAR`, so the test set is 4x the size of the
    base CIFAR test split. Each underlying image therefore appears
    exactly once at each of the rotations in ``rotation_degrees``;
    accuracy reported by the runner is the average top-1 across the
    four rotated copies (rotation-equivariance-aware accuracy).

    Returns
    -------
    (train_loader, test_loader, num_classes, in_channels)
        Standard quadruple compatible with the runner.
    """
    root = Path(root)
    if variant == "cifar10":
        Ds = CIFAR10
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        n_cls = 10
    elif variant == "cifar100":
        Ds = CIFAR100
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        n_cls = 100
    else:
        raise ValueError(variant)

    # Train: standard CIFAR augmentations + stochastic rotation over the
    # angle set. RandomRotation accepts a (min, max) range; for a
    # discrete-cardinal-angle set we sample uniformly via a Lambda.
    angles = tuple(int(a) for a in rotation_degrees)
    fill_list = [float(fill)] * 3

    def _random_cardinal_rotate(img):
        """Pick one of the discrete angles uniformly at random."""
        a = int(angles[torch.randint(0, len(angles), (1,)).item()])
        if a == 0:
            return img
        return T.functional.rotate(img, a, fill=fill_list)

    train_tfs = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.Lambda(_random_cardinal_rotate),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tfs = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    tr = Ds(root=str(root), train=True, download=True, transform=train_tfs)
    te_base = Ds(root=str(root), train=False, download=True, transform=eval_tfs)
    te = _RotatedCIFAR(te_base, angles=angles, fill=fill)

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True,
                           drop_last=True,
                           persistent_workers=num_workers > 0)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=num_workers > 0)
    return tr_loader, te_loader, n_cls, 3


def load_rotated_cifar10(root: str = "./data", batch_size: int = 256,
                         num_workers: int = 0,
                         rotation_degrees: tuple[int, ...] = (0, 90, 180, 270),
                         fill: int = 0):
    """Backwards-compatible alias used by tests and the cfg routing.

    Forwards directly to :func:`rotated_cifar_loaders` with
    ``variant='cifar10'``.
    """
    return rotated_cifar_loaders(
        root=root, batch_size=batch_size, num_workers=num_workers,
        variant="cifar10", rotation_degrees=rotation_degrees, fill=fill,
    )


def _download_imagenette_tgz(dest_tgz: Path) -> None:
    """Fetch ``imagenette2-160.tgz`` to ``dest_tgz``.

    Tries the default ``urllib.request.urlretrieve`` first; if SSL
    verification fails (the corp-cert / MITM scenario flagged in
    CLAUDE.md §2 — Python 3.13 + Windows + corporate-proxy), retries
    once with an unverified SSL context. Network errors are
    re-raised so the caller can decide to abort vs continue with a
    helpful message.
    """
    dest_tgz.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(IMAGENETTE_URL, dest_tgz)
        return
    except (ssl.SSLError, urllib.error.URLError) as e:
        # CLAUDE.md §2 corp-cert workaround: retry once with an
        # unverified SSL context. We still verify the extracted
        # directory layout after extraction, so a MITM injection that
        # corrupts the tarball would fail at extract / ImageFolder
        # time rather than silently succeed.
        warnings.warn(
            f"[imagenette] SSL/URL error on initial download ({e!r}); "
            f"retrying with unverified SSL context (corp-cert fallback).",
            stacklevel=2,
        )
        ctx = ssl._create_unverified_context()
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ctx)
        )
        with opener.open(IMAGENETTE_URL) as resp, open(dest_tgz, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)


def _ensure_imagenette_extracted(root: Path) -> Path:
    """Ensure ``<root>/imagenette2-160/`` exists; download + extract if not.

    Returns the absolute path to the extracted dataset root containing
    ``train/`` and ``val/`` subdirectories. Idempotent: if the
    extracted layout is already present, no download or extraction is
    performed. If the tarball is present but extraction failed
    half-way (no ``train/`` dir), the tarball is re-extracted.
    """
    root = Path(root)
    ds_dir = root / IMAGENETTE_DIRNAME
    train_dir = ds_dir / "train"
    val_dir = ds_dir / "val"
    if train_dir.is_dir() and val_dir.is_dir():
        return ds_dir
    tgz_path = root / IMAGENETTE_TGZ_NAME
    if not tgz_path.is_file():
        print(
            f"[imagenette] downloading {IMAGENETTE_URL} -> {tgz_path} "
            f"(~98 MB); first-run only.",
            file=sys.stderr,
        )
        _download_imagenette_tgz(tgz_path)
    print(
        f"[imagenette] extracting {tgz_path.name} -> {root} ...",
        file=sys.stderr,
    )
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extractall(path=str(root))
    if not (train_dir.is_dir() and val_dir.is_dir()):
        raise RuntimeError(
            f"Imagenette extraction failed: expected {train_dir} and "
            f"{val_dir} after extracting {tgz_path}. Delete the .tgz "
            f"and re-run to retry."
        )
    return ds_dir


def _imagenette_tfs(image_size: int, train: bool,
                    randaugment_n: int = 0, randaugment_m: int = 9,
                    random_erasing_p: float = 0.0):
    """Build the Imagenette train/test transform pipeline.

    The train pipeline is:
        RandomResizedCrop(image_size) → HFlip → [RandAugment]
        → ToTensor → Normalize(ImageNet) → [RandomErasing]

    The eval pipeline is:
        Resize(image_size + 32) → CenterCrop(image_size)
        → ToTensor → Normalize(ImageNet)

    The ``randaugment_n`` / ``random_erasing_p`` knobs mirror the
    CIFAR loader's modern-recipe entry points so the same YAML keys
    drive both datasets. Default ``randaugment_n=0`` /
    ``random_erasing_p=0.0`` reproduces the legacy
    RandomResizedCrop + HFlip baseline.
    """
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    if train:
        steps: list = [
            T.RandomResizedCrop(int(image_size)),
            T.RandomHorizontalFlip(),
        ]
        if randaugment_n and randaugment_n > 0:
            from .randaugment import build_randaugment
            steps.append(build_randaugment(num_ops=int(randaugment_n),
                                           magnitude=int(randaugment_m)))
        steps.extend([T.ToTensor(), T.Normalize(mean, std)])
        if random_erasing_p and random_erasing_p > 0.0:
            from .random_erasing import build_random_erasing
            steps.append(build_random_erasing(p=float(random_erasing_p)))
        return T.Compose(steps)
    return T.Compose([
        T.Resize(int(image_size) + 32),
        T.CenterCrop(int(image_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def imagenette_loaders(root: str = "./data", batch_size: int = 128,
                       num_workers: int = 0, image_size: int = 160,
                       randaugment_n: int = 0, randaugment_m: int = 9,
                       random_erasing_p: float = 0.0,
                       headline_mode: bool = False, seed: int = 0):
    """Load Imagenette v2-160 (10-class ImageNet subset).

    Downloads ``imagenette2-160.tgz`` on first call if not already
    extracted under ``<root>/imagenette2-160/`` (~98 MB; ~13 k train
    + ~3.9 k val images). Subsequent calls are cache-only.

    The augmentation pipeline mirrors the CIFAR loader's modern-recipe
    entry points so the same Wave-0 YAML knobs (`randaugment_n`,
    `randaugment_m`, `random_erasing_p`) drive both datasets.

    The ``headline_mode`` flag installs a deterministic per-loader
    ``torch.Generator`` + ``worker_init_fn=seed_worker`` so that
    headline / cert runs (Synthesis-100 D4) are bit-reproducible
    across re-runs at the same ``seed``.

    Parameters
    ----------
    root
        Directory under which ``imagenette2-160/`` lives (cache root).
    batch_size
        Batch size; default 128 because 160x160 is ~25x more pixels
        than CIFAR's 32x32 and the laptop 4090 16 GB VRAM tops out
        around batch 128 for ResNet-20 at bf16 (CLAUDE.md §2).
    num_workers
        DataLoader worker count; default 0 per the Windows hardware
        contract (CLAUDE.md §2).
    image_size
        Train crop side and eval centre-crop side. Default 160 to
        match the source tarball's native short-edge resize.
    randaugment_n / randaugment_m
        RandAugment configuration (default off). When ``n>0``, a
        RandAugment is inserted between HFlip and ToTensor.
    random_erasing_p
        Random Erasing probability (default off). When ``> 0``, a
        RandomErasing is appended after Normalize.
    headline_mode
        Synthesis-100 D4: when True, install deterministic
        worker_init_fn + per-loader generator.
    seed
        Per-run seed for the deterministic DataLoader generator.

    Returns
    -------
    (train_loader, test_loader, num_classes, in_channels)
        ``num_classes == 10`` always (Imagenette has 10 ImageNet
        synsets). ``in_channels == 3``.
    """
    root_path = Path(root)
    ds_dir = _ensure_imagenette_extracted(root_path)
    train_tfs = _imagenette_tfs(image_size, train=True,
                                randaugment_n=randaugment_n,
                                randaugment_m=randaugment_m,
                                random_erasing_p=random_erasing_p)
    eval_tfs = _imagenette_tfs(image_size, train=False)
    train_set = ImageFolder(str(ds_dir / "train"), transform=train_tfs)
    eval_set = ImageFolder(str(ds_dir / "val"), transform=eval_tfs)
    n_cls = len(train_set.classes)
    if n_cls != 10:
        warnings.warn(
            f"[imagenette] expected 10 classes; found {n_cls} at "
            f"{ds_dir}/train. Continuing anyway.",
            stacklevel=2,
        )
    train_kwargs: dict = dict(
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    if headline_mode:
        from .runner import seed_worker
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        train_kwargs["generator"] = gen
        train_kwargs["worker_init_fn"] = seed_worker
    tr_loader = DataLoader(train_set, **train_kwargs)
    te_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=num_workers > 0)
    return tr_loader, te_loader, n_cls, 3


def load_dataset(name: str, root: str = "./data", batch_size: int = 256,
                 num_workers: int = 4,
                 randaugment_n: int = 0, randaugment_m: int = 14,
                 random_erasing_p: float = 0.0,
                 image_size: int = 160,
                 headline_mode: bool = False, seed: int = 0):
    name = name.lower()
    if name in {"cifar10", "cifar100"}:
        return cifar_loaders(root=root, batch_size=batch_size,
                             num_workers=num_workers, variant=name,
                             randaugment_n=randaugment_n,
                             randaugment_m=randaugment_m,
                             random_erasing_p=random_erasing_p,
                             headline_mode=headline_mode, seed=seed)
    if name in {"rotated_cifar10", "rotcifar10"}:
        return rotated_cifar_loaders(
            root=root, batch_size=batch_size,
            num_workers=num_workers, variant="cifar10",
        )
    if name in {"rotated_cifar100", "rotcifar100"}:
        return rotated_cifar_loaders(
            root=root, batch_size=batch_size,
            num_workers=num_workers, variant="cifar100",
        )
    if name in {"imagenette", "imagenette2", "imagenette2-160"}:
        return imagenette_loaders(
            root=root, batch_size=batch_size, num_workers=num_workers,
            image_size=image_size,
            randaugment_n=randaugment_n,
            randaugment_m=randaugment_m,
            random_erasing_p=random_erasing_p,
            headline_mode=headline_mode, seed=seed,
        )
    if name.startswith("medmnist:"):
        flag = name.split(":", 1)[1]
        return medmnist_loaders(root=f"{root}/medmnist", flag=flag,
                                batch_size=batch_size, num_workers=num_workers)
    raise ValueError(f"unknown dataset '{name}'")

"""H56 — Cymatic Pattern Synthetic Dataset (Chladni eigenmodes).

Design doc: ``hypotheses/g6_topological_bridging/H56_cymatic_pattern_dataset.md``.

Generates a synthetic image dataset of Chladni plate-vibration patterns
labelled by their ``(m, n)`` Laplacian eigenmode index. The dataset is
constructed analytically from the steady-state 2D wave-equation
solution

    u_{m, n}(x, y) = sin(m * pi * x / h) * sin(n * pi * y / w)

so it is fully deterministic given a seed. The classes are the
``(m, n)`` mode pairs with ``2 <= m, n <= 5`` -- a 16-class
controlled-resonance classification benchmark designed as the natural-
fit task for cymatic-init (H35) and cymatic-loss (H46) priors.

Public surface
--------------
- :func:`generate_cymatic_pattern` -- ``(h, w)`` float32 tensor in
                                       ``[-1, 1]`` for a single mode.
- :class:`CymaticDataset`          -- ``torch.utils.data.Dataset`` of
                                       ``(image, class_label)`` pairs.
- :func:`generate_dataset`         -- ``TensorDataset`` compatible
                                       with ``data.load_dataset``.

References (Citation Rigor)
---------------------------
    Chladni, Ernst F. F. 1787 (Leipzig) 'Entdeckungen ueber die
    Theorie des Klanges' -- the 1787 origin of cymatic patterns;
    foundational reference.
    Rahaman, Nasim and others 2019 ICML 'On the Spectral Bias of
    Neural Networks' (arXiv:1806.08734) -- spectral bias; relevant
    to the eigenmode-classification motivation.
"""
from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset, TensorDataset


# Class table: all (m, n) mode pairs with 2 <= m, n <= 5.
# -> 4 * 4 = 16 distinct modes.
_MODE_MIN = 2
_MODE_MAX = 5
_N_MODES_PER_AXIS = _MODE_MAX - _MODE_MIN + 1
_N_CLASSES = _N_MODES_PER_AXIS * _N_MODES_PER_AXIS  # 16


def _class_to_mode(cls: int) -> tuple[int, int]:
    """Map class index in ``[0, 16)`` to ``(m, n)`` mode pair."""
    if not (0 <= cls < _N_CLASSES):
        raise ValueError(f"class={cls} out of range [0, {_N_CLASSES})")
    m = _MODE_MIN + cls // _N_MODES_PER_AXIS
    n = _MODE_MIN + cls % _N_MODES_PER_AXIS
    return m, n


def _mode_to_class(m: int, n: int) -> int:
    if not (_MODE_MIN <= m <= _MODE_MAX and _MODE_MIN <= n <= _MODE_MAX):
        raise ValueError(
            f"(m, n)=({m}, {n}) outside supported range "
            f"[{_MODE_MIN}, {_MODE_MAX}]^2"
        )
    return (m - _MODE_MIN) * _N_MODES_PER_AXIS + (n - _MODE_MIN)


def generate_cymatic_pattern(
    h: int = 32,
    w: int = 32,
    mode: tuple[int, int] = (2, 3),
    freq: float = 1.0,
    decay: float = 0.0,
) -> torch.Tensor:
    """Return the steady-state ``(m, n)`` Chladni eigenmode on an ``h x w`` grid.

    Solves the 2D wave equation ``Laplacian u + lambda u = 0`` on a
    rectangular plate with Dirichlet boundary, then evaluates at the
    pixel centres of an ``h x w`` grid. The output is normalised to
    ``[-1, 1]`` (the analytical formula is already bounded by 1 in
    magnitude, but a small numerical clamp is applied).

    Parameters
    ----------
    h, w : int
        spatial size in pixels.
    mode : tuple[int, int]
        ``(m, n)`` eigenmode index. ``m, n >= 1`` for the formula to
        be non-trivial.
    freq : float, default 1.0
        multiplicative frequency knob -- scales the spatial argument
        of ``sin``. ``freq=1`` reproduces the canonical eigenmode.
    decay : float, default 0.0
        radial decay coefficient. ``decay=0`` returns the bare
        eigenmode; non-zero values multiply by
        ``exp(-decay * r_from_centre)`` to model damped excitation.

    Returns
    -------
    torch.Tensor, shape ``(h, w)``, dtype float32, values in [-1, 1].
    """
    m, n = mode
    if m < 1 or n < 1:
        raise ValueError(f"mode=({m}, {n}) must satisfy m>=1, n>=1")
    # Cell-centred grid in [0, 1] so the boundary is at x=0, x=1 (and
    # similarly for y). Using ``arange + 0.5`` keeps the boundary
    # outside the visible pixel centres, which avoids exact-zero
    # rows/columns at the edges.
    x = (torch.arange(w, dtype=torch.float32) + 0.5) / w
    y = (torch.arange(h, dtype=torch.float32) + 0.5) / h
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")  # (h, w)
    u = (
        torch.sin(m * math.pi * grid_x * freq)
        * torch.sin(n * math.pi * grid_y * freq)
    )
    if decay > 0.0:
        cx, cy = 0.5, 0.5
        r = ((grid_x - cx) ** 2 + (grid_y - cy) ** 2).sqrt()
        u = u * torch.exp(-decay * r)
    # Numerical safety: clamp to [-1, 1].
    u = u.clamp(-1.0, 1.0)
    return u


def generate_dataset(
    n_samples: int = 1000,
    h: int = 32,
    w: int = 32,
    seed: int = 0,
) -> TensorDataset:
    """Generate a balanced ``TensorDataset`` of Chladni patterns.

    Each sample picks a class uniformly from the 16 mode classes,
    generates the corresponding eigenmode pattern, applies a small
    Gaussian perturbation (seeded), and packages the result as a
    ``(1, h, w)`` float32 image with the integer class label.

    Parameters
    ----------
    n_samples : int
        total number of samples in the resulting dataset.
    h, w : int
        image height / width.
    seed : int
        RNG seed for the per-sample class draw and additive noise.

    Returns
    -------
    torch.utils.data.TensorDataset
        ``(images, labels)`` where ``images`` is shape
        ``(n_samples, 1, h, w)`` float32 and ``labels`` is shape
        ``(n_samples,)`` int64.
    """
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, _N_CLASSES, (n_samples,), generator=g, dtype=torch.int64)
    images = torch.empty(n_samples, 1, h, w, dtype=torch.float32)
    for i in range(n_samples):
        cls = int(labels[i].item())
        m, n = _class_to_mode(cls)
        pat = generate_cymatic_pattern(h=h, w=w, mode=(m, n))
        # additive Gaussian noise, deterministic given the seed
        noise = torch.randn((h, w), generator=g) * 0.02
        images[i, 0] = (pat + noise).clamp(-1.0, 1.0)
    return TensorDataset(images, labels)


class CymaticDataset(Dataset):
    """``Dataset`` of ``(image, class_label)`` Chladni-pattern pairs.

    Lazy variant of :func:`generate_dataset`; generation happens at
    ``__getitem__`` time using a deterministic per-index seed so the
    dataset is reproducible without materialising all samples in RAM.

    Parameters
    ----------
    n_samples : int
        number of samples reported by ``len()``.
    h, w : int
        image size.
    seed : int
        base RNG seed; sample ``i`` uses ``seed + i`` as its draw.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        h: int = 32,
        w: int = 32,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if n_samples <= 0:
            raise ValueError(f"n_samples must be > 0; got {n_samples}")
        self.n_samples = int(n_samples)
        self.h = int(h)
        self.w = int(w)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if not (0 <= idx < self.n_samples):
            raise IndexError(f"idx={idx} out of range [0, {self.n_samples})")
        g = torch.Generator().manual_seed(self.seed + idx)
        cls = int(torch.randint(0, _N_CLASSES, (1,), generator=g).item())
        m, n = _class_to_mode(cls)
        pat = generate_cymatic_pattern(h=self.h, w=self.w, mode=(m, n))
        noise = torch.randn((self.h, self.w), generator=g) * 0.02
        img = (pat + noise).clamp(-1.0, 1.0).unsqueeze(0)  # (1, h, w)
        return img, cls

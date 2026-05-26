# H28 — Cymatic Hex Resonance

> **One-line claim:** Dynamically modulating the seven hex-stencil tap weights with a per-channel φ-harmonic oscillator (cymatic resonance) raises top-1 on time-varying / spatiotemporal datasets and improves convergence stability versus a static hex baseline of matched parameter budget.
>
> **Source design space:** G3 Topologies & Graphs (H21–H30).
>
> **Implementation status (this repo):** `○ not started` (compositional; builds on H21 and H35).

This document is the committee-grade design write-up for hypothesis H28.

---

## 1. Motivation (sacred / natural intuition, ≥ 100 words)

Cymatics — Ernst Chladni's 1787 demonstration that vibrating plates form geometric patterns at resonance frequencies — is nature's mechanism for self-organization of geometry from vibration. The patterns are eigenmodes of the wave equation on the plate domain; their shapes depend only on the plate boundary and the driving frequency. At a hex-lattice resonance, the eigenmodes form hexagonal node patterns precisely because the hex lattice is the natural 2-D vibration basis. Modulating the hex stencil's tap weights with a φ-harmonic temporal oscillator (i.e., `w_neighbour(t) = w_0 · cos(ω·t + φ·k)`) layers in **temporal cymatic resonance** on top of the spatial hex prior — a "vibrating honeycomb" filter.

In deep learning, such dynamic modulation is uncommon but motivated: (a) for **spatiotemporal data** (video, weather, fMRI), the temporal axis is real and a phase-locked filter exploits it; (b) for **training dynamics**, periodic modulation of weights acts like a Lyapunov-stabilization mechanism that prevents weight collapse to trivial solutions (Sussillo & Abbott 2009 on neural dynamics); (c) for **interpretability**, the φ-harmonic phases form an explicit "frequency band" structure that maps to known cortical theta/alpha/gamma rhythms. Combined with H21's hex prior, the resulting "cymatic hex resonance" filter is the simplest dynamic geometric prior our research program can field.

## 2. Formal hypothesis (the falsifiable claim, ≥ 50 words)

We claim that **because** hex resonance is a wave-equation eigenmode and φ-harmonics are the most-irrational frequency ratios (avoiding interference), modulating the 7 hex-stencil taps with `w_k(t) = w_0,k · cos(ω·t + k·φ)` raises top-1 on UCF-101 video subset by ≥ +2 pp and reduces convergence variance across seeds by ≥ 30 %, per the mechanism of Chladni 1787, Sussillo & Abbott 2009 (arXiv:0903.4537), and our own H35 cymatic-init lesson (T1.7 negative — see § 11).

## 3. Falsifier (≥ 30 words)

If, at 3-seed median on UCF-101 (subset, 25-class), the cymatic-hex-resonance variant fails to lift top-1 by ≥ 1.0 pp AND fails to reduce seed-to-seed convergence variance by ≥ 15 %, this hypothesis is **DISCARDED**.

## 4. Citations (Citation Rigor format, ≥ 80 words)

```
Chladni, E. 1787 'Entdeckungen über die Theorie des Klanges' — the
original cymatic-pattern reference; sound + geometry on plates.

Sussillo, D., Abbott, L. F. 2009 'Generating Coherent Patterns of
Activity from Chaotic Neural Networks' (arXiv:0903.4537) — dynamic-weight
modulation in recurrent networks; theoretical grounding for periodic
weight oscillation.

Buzsaki, G., Draguhn, A. 2004 Science 'Neuronal Oscillations in
Cortical Networks' — biological evidence for harmonic rhythm in
cortical computation.

Hoogeboom, E., et al. 2018 ICML 'HexaConv' (arXiv:1803.02108) — the
hex-stencil reference H28 modulates dynamically.

Tran, D., et al. 2018 CVPR 'A Closer Look at Spatiotemporal
Convolutions' (arXiv:1711.11248) — UCF-101 / spatiotemporal CNN
benchmark methodology.
```

## 5. Mechanism

### 5.1 CNN track

Implement `CymaticHexConv3d` (B, C, T, H, W) that combines a hex spatial mask with a per-channel φ-harmonic temporal modulation of the 6 peripheral taps.

```python
# ideas/28_cymatic_hex_resonance/implementation.py
PHI = (1+5**0.5)/2

class CymaticHexConv3d(nn.Conv3d):
    def __init__(self, *a, omega_init=0.1, **kw):
        super().__init__(*a, **kw)
        self.register_buffer("hex_mask", torch.tensor([[
            [0., 1., 1.], [1., 1., 1.], [1., 1., 0.]
        ]]).view(1, 1, 1, 3, 3))
        self.omega = nn.Parameter(torch.full((self.out_channels,), omega_init))
        # phase offsets per neighbour tied to k*phi
        phases = torch.tensor([k * PHI for k in range(6)])
        self.register_buffer("phases", phases)

    def forward(self, x):
        # x: (B, C, T, H, W); apply hex mask and time-varying scale
        T = x.size(2)
        t = torch.arange(T, device=x.device).float()
        # modulation per output-channel, per neighbour, per time
        mod = torch.cos(self.omega.unsqueeze(1) * t + self.phases.unsqueeze(0).sum(-1, keepdim=True))
        # multiply weight by modulation along time dimension and hex_mask in spatial
        w = self.weight * self.hex_mask
        # this implementation needs per-time-step convolution — omitted detail
        return F.conv3d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
```

- Params: same as 3-D hex conv plus C extra ω parameters (negligible).
- FLOPs: 7-tap hex × T time × spatial.
- Init: He init plus ω_init = 0.1 to start at slow modulation.

### 5.2 LLM track

For decoder-only Transformers, cymatic-hex-resonance maps onto **time-varying attention bias**: add a per-head `cos(ω·t + k·φ)` term to the attention logits, where `t` is token position. This is a dynamic equivalent of RoPE that introduces explicit oscillation.

- Slots in: attention logits, additive bias before softmax.
- Causal-mask preservation: ✓.
- FlashAttention-2 compatibility: requires custom kernel for time-varying bias; fall back to manual attention for prototype.

```python
class CymaticAttentionBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.omega = nn.Parameter(torch.full((heads,), 0.05))
        phases = torch.tensor([k * PHI for k in range(heads)])
        self.register_buffer("phases", phases)
    def forward(self, t_q, t_k):
        # add cos(omega * (t_q - t_k) + phase) bias to attention logits
        ...
```

Expected at 124 M: **+0.0 to +0.3 perplexity** (mild positive); training stability may improve.

## 6. Predicted Δ on 4090 benchmarks

| metric | Δ vs. baseline | rationale |
|---|---|---|
| composite (UCF-101 subset) | [+0.005, +0.025] | spatiotemporal data should reward time-varying filter |
| top-1 (UCF-101 subset, primary) | [+2.0 pp, +5.0 pp] | direct claim |
| top-1 (CIFAR-10) | [-1.5 pp, +0.5 pp] | static data; time modulation provides little benefit |
| Convergence variance across seeds | [-15 %, -40 %] | resonance acts like Lyapunov stabilization |
| params | [+0.5 %, +1 %] | added ω parameter per channel |
| FLOPs | [+5 %, +15 %] | time-varying multiply per step |
| GPU latency | [×1.2, ×1.5] | extra cos modulation each step |
| Betti collapse rate | [-0.05, +0.10] | dynamic weights destabilize topology initially |

## 7. Experimental protocol

### 7.1 Primary experiment

- **Dataset:** UCF-101 (25-class subset, 8-frame clips at 112×112).
- **Architecture:** 3-D NaturePriorBlock with `CymaticHexConv3d` (spatial hex × temporal cymatic).
- **Epochs / batch / precision / seeds:** 30 epochs, batch 16 (video), bf16 AMP, 3 seeds.
- **Composite:** top-1 (0.5), convergence variance (0.2), params (0.15), latency (0.15).
- **Run-script:** `python ideas/28_cymatic_hex_resonance/experiment.py --seeds 0 1 2`.
- **Wall-clock:** ~3 hr/seed × 3 = 9 hr.
- **Archive:** `ideas/28_cymatic_hex_resonance/experiments/exp001_ucf_seed0..2/`.

### 7.2 Idea-targeted experiment

1. **UCF-101 / Something-Something** — temporal-action recognition.
2. **Weather datasets (ERA5 subset)** — spatiotemporal with real cyclic periods.
3. **fMRI / EEG classification** — biological oscillation alignment.

### 7.3 Cross-paradigm context (LLM track)

WikiText-103 at 124 M with `CymaticAttentionBias` on half the attention heads; report perplexity vs baseline and gradient-norm variance across training.

## 8. Cross-references

- Parent: `IDEA_TABLE.md` § G3 row H28.
- Master: planned Tier 3.
- Sub-dir: `ideas/28_cymatic_hex_resonance/`.
- Composes: H21 (hex), H35 (cymatic init — see § 11 lesson), H46 (cymatic loss), H66 (cymatic QKV).
- Conflicts: H22 (toroidal — different topology); H50 (full hybrid bundled).

## 9. Committee Q&A

**Q: Why isn't this just gated weights with a sinusoidal schedule?**

> The schedule is **structured by φ phases** (each neighbour at `k·φ` phase offset), not arbitrary. The structure makes the bias falsifiable: a random-phase variant is the natural control.

**Q: T1.7 cymatic-init was negative. Why expect dynamic cymatic to help?**

> T1.7 fixed the init at one Chladni mode and trained statically. Dynamic resonance keeps the mode oscillating, which (a) avoids the "wrong-mode lock-in" failure and (b) provides Lyapunov-style stabilization (Sussillo 2009). The dynamic case is structurally different from the static init.

**Q: How is this falsifiable rather than aesthetic?**

> § 3 specifies: top-1 ≥ +1.0 pp AND seed-variance ≤ -15 %. Both must hold.

**Q: What if the prior helps on UCF but hurts on static data?**

> That is **consistent**: the prior's design regime is spatiotemporal data.

**Q: How do we know the implementation is correct?**

> `tests/test_cymatic_hex.py::test_hex_zero_corners` asserts hex mask shape. `test_phi_phases` asserts phase offsets follow `k·φ`. `test_omega_learnable` asserts ω parameter receives gradient. Plus archive verification.

## 10. Verification artifacts checklist

- [ ] `ideas/28_cymatic_hex_resonance/implementation.py` tests green
- [ ] `ideas/28_cymatic_hex_resonance/tests.py` ≥ 6 assertions
- [ ] AUDIT.md ≥ 3 weaknesses
- [ ] IMPROVEMENTS.md
- [ ] VERIFY.md signed
- [ ] Experiment archives + verification
- [ ] Row added to `EXPERIMENT_LOG.md` Tier 3
- [ ] FINDINGS reflects result

## 11. Cymatic lessons learned from T1.7 — what changes in H28

T1.7 (`sg_only_cymatic_init`) on upright CIFAR-10 yielded top-1 77.44 % vs reference 80.11 % (-2.67 pp), composite 0.7883 vs 0.8135 (-0.0252). The negative result was **unexpected** under the source PDF's prediction. Three lessons inform H28:

1. **Static cymatic init locks the filter to one Chladni mode.** Once initialized, gradient descent has no special incentive to preserve the cymatic structure; the prior is "spent at init" and quickly washed out by training. Dynamic resonance (this hypothesis) keeps the structure ALIVE through training.
2. **Static data does not reward harmonic priors.** CIFAR-10 has no temporal axis. UCF-101 / spatiotemporal data does, and that is the proper target.
3. **Single Chladni mode lacks orthogonality across channels.** All channels were initialized from the same Chladni eigenmode set, so the priors were correlated. H28 distributes phase offsets via `k·φ`, decorrelating channels by golden-ratio phase spacing.

**What H28 changes from H35:** (a) DYNAMIC modulation instead of static init; (b) phase offsets follow `k·φ` rather than identical phase; (c) target dataset is spatiotemporal not static.

## 12. Status journal

- 2026-05-27 — Created from template by Doc-Agent-B. Integrates T1.7 cymatic-init negative-result lesson into the dynamic variant.

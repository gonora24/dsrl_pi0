<div align="center">

# Temporal Latent Steering for Flow-Matching VLAs

### Master's thesis code, built on DSRL (π₀) — extended to π₀.₅ and X-VLA

**Nora Gorhan** · Karlsruhe Institute of Technology (KIT)

[Thesis: available on request] &nbsp; [[original DSRL paper](https://arxiv.org/abs/2506.15799)] &nbsp; [[original DSRL repo](https://github.com/nakamotoo/dsrl_pi0)]

</div>

> **Note: this is a fork, not the official DSRL implementation.**
> This repository is forked from [`nakamotoo/dsrl_pi0`](https://github.com/nakamotoo/dsrl_pi0), the official code of
> [*Steering Your Diffusion Policy with Latent Space Reinforcement Learning*](https://arxiv.org/abs/2506.15799) (Wagenmaker et al., CoRL 2025).
> The DSRL-SAC baseline, the training pipeline and the π₀ integration come from the original authors.
> Everything listed under [What this repository adds](#what-this-repository-adds) was developed for my master's thesis.
> Please direct questions about this fork to me (via GitHub issues), not to the original authors.

---

## Overview

DSRL steers a frozen, pretrained flow-matching VLA by running RL in its **latent-noise space**: a small actor predicts the noise that is fed to the action expert, while the VLA itself stays untouched. The original method predicts a *single* latent vector and repeats it across the whole action chunk.

This thesis investigates whether **temporally varying** latent steering (a separate latent vector per action) can improve on this repetition strategy, and how its much larger exploration space (e.g. 32 → 320 dimensions for π₀.₅) can be made tractable.

## What this repository adds

| | Original ([`dsrl_pi0`](https://github.com/nakamotoo/dsrl_pi0)) | This fork |
|---|---|---|
| **Models** | π₀ | π₀, **π₀.₅**, **X-VLA** |
| **Steering** | DSRL-SAC (repeated single latent vector) | + temporal steering methods (see below) |
| **Actor / critic** | MLP | MLP + **transformer** (autoregressive actor, causal critic) |
| **Analysis** | – | **Gradient-sensitivity analysis** of latent dimensions |
| **Evaluation** | Libero, Aloha, real Franka | Extended multi-task evaluation on **LIBERO-90** |

### Implemented methods

All numbers refer to the optimized latent dimensionality for π₀.₅ (action horizon 10, latent dimension 32).

| Method | Idea | Dim. |
|---|---|---|
| **DSRL-SAC** *(original)* | Predict one latent vector, repeat across the chunk | 32 |
| **RDS** | Repetition, but only on the ~7 most influential latent dimensions (found by gradient-sensitivity analysis) | 7 |
| **FDTS-Chunk** | Full temporal steering: one latent vector per action, optimized jointly with Q-chunking | 320 |
| **FDTS-Residual** (Noise / Distribution) | Repeated vector as starting point + autoregressive residuals (on the noise or on the distribution parameters) | 320 |
| **T-RDS** | Temporal steering restricted to the influential dimensions | 70 |
| **GT-RDS** | Grouped temporal steering: fewer distinct vectors, each repeated over a segment of the chunk | 35 |
| **Bounded GT-RDS-Residual** | GT-RDS with bounded residuals | 35 |

The remaining latent dimensions of the reduced methods (RDS, T-RDS, GT-RDS) are sampled from the standard Gaussian prior, since the flow-matching head needs a full-dimensional input.

### Supported models

| Model | Latent dim. per action | Action horizon | Checkpoint |
|---|---|---|---|
| π₀ | 32 | 50 | LIBERO-finetuned ([openpi](https://github.com/Physical-Intelligence/openpi)) |
| π₀.₅ | 32 | 10 | LIBERO-finetuned ([openpi](https://github.com/Physical-Intelligence/openpi)) |
| X-VLA | 20 | 50 | LIBERO-finetuned ([X-VLA](https://github.com/2toinf/X-VLA)) |

## Installation

Installation follows the original repository.

1. Create a conda environment:
```bash
conda create -n dsrl_pi0 python=3.11.11
conda activate dsrl_pi0
```

2. Clone this repo with all submodules:
```bash
git clone TODO-your-repo-url --recurse-submodules
cd TODO-repo-name
```

3. Install all packages and dependencies:
```bash
pip install -e .
pip install -r requirements.txt
pip install "jax[cuda12]==0.5.0"

# install openpi
pip install -e openpi
pip install -e openpi/packages/openpi-client

# install Libero
pip install -e LIBERO
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu # needed for libero

# uninstall conflicting cuda versions
pip uninstall -y $(pip list --format=freeze | grep '^nvidia-' | cut -d= -f1)

# reinstall jax
pip install "jax[cuda12]==0.5.0"

# to handle conflicting CUDA versions
unset LD_LIBRARY_PATH

# for X-VLA
pip install huggingface-hub==0.36.0
```

## Usage

> **TODO:** replace the placeholders below with your actual scripts / flags.

**Baseline (original DSRL-SAC)**
```bash
bash examples/scripts/run_libero_pi05.sh
```

**Thesis methods**

All scripts are named after their methods, please choose the appropriate bash script such as
```bash
bash examples/scripts/run_libero_pi05_chunk.sh
```

**Gradient-sensitivity analysis**
```bash
bash examples/scripts/run_sensitivity_matrix_gradients.sh
```

The original Aloha and real-robot (Franka) pipelines are kept from upstream but were **not** used or tested for the thesis.
For those, please refer to the [original repository](https://github.com/nakamotoo/dsrl_pi0).

## Citation

If you use this code, please cite **both** the original DSRL paper and the thesis.

**Original DSRL paper**
```bibtex
@article{wagenmaker2025steering,
  author    = {Andrew Wagenmaker and Mitsuhiko Nakamoto and Yunchu Zhang and Seohong Park and Waleed Yagoub and Anusha Nagabandi and Abhishek Gupta and Sergey Levine},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```

## Acknowledgements

- [**DSRL**](https://github.com/nakamotoo/dsrl_pi0) by Wagenmaker, Nakamoto et al. — the codebase this work is forked from, including the DSRL-SAC baseline
- [**jaxrl2**](https://github.com/ikostrikov/jaxrl2) and [**PTR**](https://github.com/Asap7772/PTR), on which the original repository is built
- [**openpi**](https://github.com/Physical-Intelligence/openpi) (π₀, π₀.₅), [**X-VLA**](https://github.com/2toinf/X-VLA) and [**LIBERO**](https://github.com/Lifelong-Robot-Learning/LIBERO)

<div align="center">

# MixRoute

### Rethinking Single-Distribution Training for Generalizable Neural Routing

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://neurips.cc/)
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-b31b1b.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

**Hang Yi**<sup>1</sup> · **Ziwei Huang**<sup>1</sup> · **Zhiguang Cao**<sup>1</sup> · **Yining Ma**<sup>2</sup>

<sup>1</sup>Singapore Management University &nbsp;&nbsp; <sup>2</sup>Massachusetts Institute of Technology

</div>

---

## 📢 News

- **[2026/09]** MixRoute is accepted to **NeurIPS 2026**! 🎉
- Code and benchmark will be released soon. Stay tuned!

## 🔍 Overview

Neural Combinatorial Optimization (NCO) is a promising paradigm for solving routing problems, yet its generalization across diverse data distributions remains a critical bottleneck. Existing methods typically rely on complex multi-distribution training or meta-learning schemes.

We show that the potential of **single-distribution training** has been substantially underestimated, and can be unlocked by a simple architectural inductive bias: **Mix Normalization**.

- 🎯 Trained **exclusively on uniform distributions**, it generalizes zero-shot to a wide range of unseen distributions.
- 📊 A comprehensive **benchmark** spanning **3 routing problems** (TSP, CVRP, CVRPTW) and **179 fine-grained datasets**, covering shifts in node coordinates, customer demands, capacity and time windows.

## 🚧 Code Release

The implementation, pretrained checkpoints and the full benchmark are being prepared and will be released here soon.

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{yi2026mixroute,
  title     = {MixRoute: Rethinking Single-Distribution Training for Generalizable Neural Routing},
  author    = {Yi, Hang and Huang, Ziwei and Cao, Zhiguang and Ma, Yining},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

<div align="center">

# MixRoute

### Rethinking Single-Distribution Training for Generalizable Neural Routing

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://neurips.cc/)
[![arXiv](https://img.shields.io/badge/arXiv-2611.11977-b31b1b.svg)](https://arxiv.org/abs/2611.11977)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](License.md)

**Hang Yi**<sup>1</sup> · **Ziwei Huang**<sup>1</sup> · **Yining Ma**<sup>2,✉</sup> · **Zhiguang Cao**<sup>1</sup>

<sup>1</sup>Singapore Management University &nbsp;&nbsp; <sup>2</sup>Massachusetts Institute of Technology

</div>

---

## 🏋️ Train

```bash
python TSP/POMO/train_n100.py
python CVRP/POMO/train_n100.py
python CVRPTW/POMO/train_n100.py
```

Models are trained on uniform instances of size 100. Checkpoints and logs are saved to `<PROBLEM>/POMO/result/`.

## 🧪 Test

```bash
python TSP/POMO/test_n100.py
python CVRP/POMO/test_distribution.py
python CVRPTW/POMO/test_ood_datasets.py
```

Each script evaluates the pretrained model in `pretrained/` on all test sets in `Dataset/` and prints the gap per dataset and per family. To test another checkpoint, change the model path at the top of the script.

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{yi2026mixroute,
  title     = {MixRoute: Rethinking Single-Distribution Training for Generalizable Neural Routing},
  author    = {Yi, Hang and Huang, Ziwei and Ma, Yining and Cao, Zhiguang},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```


# Test pre-trained POMO on all npz datasets under NPZ_ROOT.
#   keys: names, sizes, coords (object array of (N,2) int scaled by SCALE_XY), scale, edge_type
# Same directory has tsp_n<N>_<B>.log with LKH baseline costs.

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

NPZ_ROOT = "../../Dataset/tsp"

MODEL_CKPT_DIR = "../../pretrained/tsp"
MODEL_CKPT_EPOCH = 6500

TEST_BATCH_SIZE = 128
AUGMENTATION_ENABLE = False
AUG_FACTOR = 8


# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")     # for problem_def
sys.path.insert(0, "../..")  # for utils


# Imports

import csv
import glob
import time
from collections import defaultdict

import numpy as np
import torch

# numpy 2.x renamed numpy.core to numpy._core; unpickling an object-dtype npz saved by
# numpy 2.x under numpy 1.x then fails, so alias numpy._core to numpy.core here.
if not hasattr(np, "_core"):
    import numpy.core as _np_core
    sys.modules["numpy._core"] = _np_core
    for _sub in ("multiarray", "umath", "numeric", "_methods", "fromnumeric"):
        _mod = getattr(_np_core, _sub, None)
        if _mod is not None:
            sys.modules[f"numpy._core.{_sub}"] = _mod

from TSPEnv import TSPEnv
from TSPModel import TSPModel
from TSProblemDef import augment_xy_data_by_8_fold


# Env: load problems from a preloaded buffer instead of random sampling

class NpzTSPEnv(TSPEnv):
    def set_buffer(self, problems_tensor: torch.Tensor):
        self._buffer = problems_tensor  # (total, N, 2) on device
        self._cursor = 0

    def reset_cursor(self):
        self._cursor = 0

    def load_problems(self, batch_size, aug_factor=1):
        end = self._cursor + batch_size
        self.problems = self._buffer[self._cursor:end]
        self._cursor = end
        self.batch_size = batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
            else:
                raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)


# Helpers

def discover_npz(root):
    return sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))


def family_of(rel_path):
    """TSP NPZ tree is `distribution/<family>/<variant>/tsp_n*.npz`,
    so family is the second path component under NPZ_ROOT."""
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[0] == "distribution":
        return parts[1]
    return parts[0] if parts else "other"


def _fmt_time(sec):
    if sec < 60:
        return f"{sec:.2f}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{int(m)}m{s:.1f}s"
    h, m = divmod(int(m), 60)
    return f"{h}h{int(m)}m{s:.1f}s"


def load_npz_to_tensor(npz_path, device):
    data = np.load(npz_path, allow_pickle=True)
    coords_obj = data["coords"]
    sizes = data["sizes"]
    scale = int(data["scale"])

    n0 = int(sizes[0])
    if not all(int(s) == n0 for s in sizes):
        raise ValueError(f"mixed sizes in {npz_path}")

    arr = np.stack([c.astype(np.float32) / scale for c in coords_obj], axis=0)  # (B, N, 2) ∈ [0,1]
    return torch.from_numpy(arr).to(device), n0, scale


def parse_lkh_log(log_path):
    """Return mean LKH cost (in scaled int units) over OK rows, or None."""
    if not os.path.exists(log_path):
        return None
    costs = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            cost_str, status = parts[4], parts[5]
            if status.startswith("OK") and cost_str != "NA":
                try:
                    costs.append(int(cost_str))
                except ValueError:
                    pass
    return float(np.mean(costs)) if costs else None


def run_on_buffer(model, env, batch_size, aug_factor):
    """Roll out POMO over env's buffer, return (no_aug_mean, aug_mean) tour lengths."""
    env.reset_cursor()
    total = env._buffer.shape[0]

    no_aug_scores, aug_scores = [], []
    model.eval()
    with torch.no_grad():
        cursor = 0
        while cursor < total:
            bs = min(batch_size, total - cursor)
            env.load_problems(bs, aug_factor)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)

            state, reward, done = env.pre_step()
            while not done:
                selected, _ = model(state)
                state, reward, done = env.step(selected)
            # reward.shape: (aug*bs, pomo)
            aug_reward = reward.reshape(aug_factor, bs, env.pomo_size)
            max_pomo, _ = aug_reward.max(dim=2)        # (aug, bs)
            no_aug_scores.append(-max_pomo[0].float())  # (bs,)
            max_aug, _ = max_pomo.max(dim=0)            # (bs,)
            aug_scores.append(-max_aug.float())
            cursor += bs

    return torch.cat(no_aug_scores).mean().item(), torch.cat(aug_scores).mean().item()


# Main

def main():
    if USE_CUDA:
        torch.cuda.set_device(CUDA_DEVICE_NUM)
        device = torch.device("cuda", CUDA_DEVICE_NUM)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    model_params = {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** (1 / 2),
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "eval_type": "argmax",
    }
    model = TSPModel(**model_params)
    ckpt_path = f"{MODEL_CKPT_DIR}/checkpoint-{MODEL_CKPT_EPOCH}.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    npz_files = discover_npz(NPZ_ROOT)
    print(f"Found {len(npz_files)} npz files under {NPZ_ROOT}")
    print(f"AUG = {AUGMENTATION_ENABLE} (factor={AUG_FACTOR if AUGMENTATION_ENABLE else 1})")
    print("=" * 80)
    aug_factor = AUG_FACTOR if AUGMENTATION_ENABLE else 1

    results = []

    family_time  = defaultdict(float)
    family_count = defaultdict(int)
    family_inst  = defaultdict(int)

    overall_t0 = time.perf_counter()

    for i, npz in enumerate(npz_files, start=1):
        rel = os.path.relpath(npz, NPZ_ROOT)
        fam = family_of(rel)

        t0 = time.perf_counter()

        coords, n_size, scale = load_npz_to_tensor(npz, device)
        B = coords.shape[0]

        env = NpzTSPEnv(problem_size=n_size, pomo_size=n_size)
        env.set_buffer(coords)

        no_aug, with_aug = run_on_buffer(model, env, TEST_BATCH_SIZE, aug_factor)

        log_path = npz[:-4] + ".log"
        lkh_scaled = parse_lkh_log(log_path)
        lkh_mean = (lkh_scaled / scale) if lkh_scaled is not None else None
        gap = ((no_aug - lkh_mean) / lkh_mean * 100) if lkh_mean else None
        gap_aug = ((with_aug - lkh_mean) / lkh_mean * 100) if lkh_mean else None

        elapsed = time.perf_counter() - t0
        family_time[fam]  += elapsed
        family_count[fam] += 1
        family_inst[fam]  += B

        gap_str = f"{gap:+.2f}%" if gap is not None else "NA"
        lkh_disp = f"{lkh_mean:.4f}" if lkh_mean is not None else "NA"
        print(f"[{i:>2}/{len(npz_files)}] {rel:60s} (fam={fam})  N={n_size} B={B}  "
              f"model={no_aug:.4f}  lkh={lkh_disp}  gap={gap_str}  "
              f"time={_fmt_time(elapsed)}")

        results.append({
            "path": rel, "fam": fam, "N": n_size, "B": B,
            "lkh_mean": lkh_mean,
            "model_no_aug": no_aug, "model_aug": with_aug,
            "gap_pct": gap, "gap_pct_aug": gap_aug,
            "time_sec": elapsed,
        })

    overall = time.perf_counter() - overall_t0

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    header = (f"{'family/variant':<60s} {'fam':>10s} {'N':>4s} {'B':>4s} "
              f"{'LKH':>10s} {'Model':>10s} {'Gap%':>8s} {'time':>10s}")
    print(header)
    print("-" * len(header))
    for r in results:
        lkh_s = f"{r['lkh_mean']:.4f}" if r["lkh_mean"] is not None else "NA"
        gap_s = f"{r['gap_pct']:+.2f}" if r["gap_pct"] is not None else "NA"
        t_s   = _fmt_time(r["time_sec"])
        path_s = r["path"]
        if len(path_s) > 60:
            path_s = "..." + path_s[-57:]
        print(f"{path_s:<60s} {r['fam']:>10s} {r['N']:>4d} {r['B']:>4d} "
              f"{lkh_s:>10s} {r['model_no_aug']:>10.4f} {gap_s:>8s} {t_s:>10s}")

    # ---- families in the order they appear in npz_files ----
    seen, ordered = set(), []
    for npz in npz_files:
        rel = os.path.relpath(npz, NPZ_ROOT)
        f = family_of(rel)
        if f not in seen:
            seen.add(f)
            ordered.append(f)

    print("\n" + "=" * 80)
    print("Per-family time")
    print("=" * 80)
    print(f"{'family':<14s} {'#npz':>5s} {'#inst':>7s} "
          f"{'total_time':>14s} {'avg/npz':>12s}")
    for fam in ordered:
        if family_count[fam] == 0:
            continue
        n_npz = family_count[fam]
        n_ins = family_inst[fam]
        t_tot = family_time[fam]
        t_avg = t_tot / n_npz
        print(f"{fam:<14s} {n_npz:5d} {n_ins:7d} "
              f"{_fmt_time(t_tot):>14s} {_fmt_time(t_avg):>12s}")

    # ---- Paper Table 4 (TSP row) ----
    # family gap = sum(model mean) / sum(LKH mean) - 1 over the family's variants (pooled);
    # Avg = arithmetic mean of the family columns (incl. Uniform).
    print("\n" + "=" * 80)
    print("Paper Table 4 (TSP): pooled gap per family")
    print("=" * 80)
    paper_cols = []
    for fam in ordered:
        recs = [r for r in results if r["fam"] == fam and r["lkh_mean"]]
        if not recs:
            continue
        g = (sum(r["model_no_aug"] for r in recs) / sum(r["lkh_mean"] for r in recs) - 1.0) * 100.0
        paper_cols.append(g)
        print(f"{fam:<14s} gap={g:+.2f}%  (n_variants={len(recs)})")
    if paper_cols:
        print(f"{'Avg':<14s} gap={float(np.mean(paper_cols)):+.2f}%  (mean of {len(paper_cols)} columns)")

    print("\n" + "=" * 80)
    print(f"OVERALL TOTAL TIME : {_fmt_time(overall)}   "
          f"({sum(family_count.values())} npz, "
          f"{sum(family_inst.values())} instances)")
    print("=" * 80)

    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_npz_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "path", "fam", "N", "B",
            "lkh_mean", "model_no_aug", "model_aug",
            "gap_pct", "gap_pct_aug", "time_sec",
        ])
        w.writeheader()
        w.writerows(results)
    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    main()

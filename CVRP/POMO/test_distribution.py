# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


# Path Config

import os
import sys
import time
import numpy as np
from collections import defaultdict
from pathlib import Path
from copy import deepcopy


os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


# import

import logging
from utils.utils import create_logger, copy_all_src

from CVRPTestercoordinate import CVRPTester as Tester


# parameters
problem_size = 100


env_params = {
    'problem_size': problem_size,
    'pomo_size': problem_size,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}


tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': '../../pretrained/cvrp',  # directory path of pre-trained model
        'epoch': 6500,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 128,
    'test_batch_size': 64,
    'augmentation_enable': False,
    'aug_factor': 8,
    'aug_batch_size': 400,
    'test_data_load': {
        'enable': True,
        'filename': None,  # filled per npz at run time
    },
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'test_cvrp100',
        'filename': 'log.txt'
    }
}


def run_one(npz_path: Path):
    # copy the global configs into local variables
    env_params = deepcopy(globals()['env_params'])
    tester_params = deepcopy(globals()['tester_params'])
    tester_params['test_data_load'] = deepcopy(tester_params['test_data_load'])

    # set the current npz
    tester_params['test_data_load']['filename'] = str(npz_path)

    # distance-matrix npz cannot use 8x geometric augmentation
    tester_params['augmentation_enable'] = False
    tester_params['test_batch_size'] = min(tester_params['test_batch_size'], 1024)

    # separate log dir per npz
    log_desc = f"test_npz__{npz_path.stem}"
    logger_params = {'log_file': {'desc': log_desc, 'filename': 'log.txt'}}
    create_logger(**logger_params)
    logger = logging.getLogger('root')
    logger.info(f"== Testing NPZ: {npz_path} ==")

    tester = Tester(env_params=env_params,
                    model_params=globals()['model_params'],
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)
    result = tester.run()
    logger.info(f"== Done: {npz_path} ==")
    return result[0] if result else None  # no-aug score


NPZ_PATHS = [
    # in-distribution baseline
    ('in_dist', '../../Dataset/cvrp/in_distribution/uniform/cvrp_n100_128.npz'),
    # demand
    ('uniform', '../../Dataset/cvrp/distribution/demand/1/out/cvrp_n100_1000.npz'),
    ('small',   '../../Dataset/cvrp/distribution/demand/2/out/cvrp_n100_1000.npz'),
    ('large',   '../../Dataset/cvrp/distribution/demand/3/out/cvrp_n100_1000.npz'),
    ('ushape',  '../../Dataset/cvrp/distribution/demand/4/out/cvrp_n100_1000.npz'),
    ('bell',    '../../Dataset/cvrp/distribution/demand/5/out/cvrp_n100_1000.npz'),
    ('bimodal', '../../Dataset/cvrp/distribution/demand/6/out/cvrp_n100_1000.npz'),
    # beta
    ('vedge',   '../../Dataset/cvrp/distribution/beta/0.1_0.1/cvrp_n100_1000.npz'),
    ('edge',    '../../Dataset/cvrp/distribution/beta/0.3_0.3/cvrp_n100_1000.npz'),
    ('vcorner', '../../Dataset/cvrp/distribution/beta/0.3_1.7/cvrp_n100_1000.npz'),
    ('corner',  '../../Dataset/cvrp/distribution/beta/0.6_1.4/cvrp_n100_1000.npz'),
    ('center',  '../../Dataset/cvrp/distribution/beta/2.0_2.0/cvrp_n100_1000.npz'),
    ('vcenter', '../../Dataset/cvrp/distribution/beta/4.0_4.0/cvrp_n100_1000.npz'),
    # gaussian
    ('g01',     '../../Dataset/cvrp/distribution/gaussian/single/0.1/cvrp_n100_1000.npz'),
    ('g02',     '../../Dataset/cvrp/distribution/gaussian/single/0.2/cvrp_n100_1000.npz'),
    ('g03',     '../../Dataset/cvrp/distribution/gaussian/single/0.3/cvrp_n100_1000.npz'),
    ('g04',     '../../Dataset/cvrp/distribution/gaussian/single/0.4/cvrp_n100_1000.npz'),
    # grid
    ('regular', '../../Dataset/cvrp/distribution/grid/grid/cvrp_n100_1000.npz'),
    ('stretch', '../../Dataset/cvrp/distribution/grid/grid_aniso/cvrp_n100_1000.npz'),
    ('broken',  '../../Dataset/cvrp/distribution/grid/grid_broken/cvrp_n100_1000.npz'),
    ('noisy',   '../../Dataset/cvrp/distribution/grid/grid_gauss/cvrp_n100_1000.npz'),
    ('rotated', '../../Dataset/cvrp/distribution/grid/grid_rotate/cvrp_n100_1000.npz'),
    ('warped',  '../../Dataset/cvrp/distribution/grid/grid_warp/cvrp_n100_1000.npz'),
    # stripes (8 one-word labels, one per layout)
    ('rails',  '../../Dataset/cvrp/distribution/stripes/1/cvrp_n100_1000.npz'),  # thin K=2 uniform: two rails
    ('solo',   '../../Dataset/cvrp/distribution/stripes/2/cvrp_n100_1000.npz'),  # thin K=2 biased: one main line
    ('comb',   '../../Dataset/cvrp/distribution/stripes/3/cvrp_n100_1000.npz'),  # thin K=8 uniform: 8-tooth comb
    ('spike',  '../../Dataset/cvrp/distribution/stripes/4/cvrp_n100_1000.npz'),  # thin K=8 biased: one dominant spike
    ('bands',  '../../Dataset/cvrp/distribution/stripes/5/cvrp_n100_1000.npz'),  # thick K=2 uniform: two parallel wide bands
    ('mono',   '../../Dataset/cvrp/distribution/stripes/6/cvrp_n100_1000.npz'),  # thick K=2 biased: one main wide band
    ('flood',  '../../Dataset/cvrp/distribution/stripes/7/cvrp_n100_1000.npz'),  # thick K=8 uniform: 8 wide bands, nearly filled
    ('core',   '../../Dataset/cvrp/distribution/stripes/8/cvrp_n100_1000.npz'),  # thick K=8 biased: one core band + scatter
    # cluster
    ('islands',     '../../Dataset/cvrp/distribution/cluster/1/cvrp_n100_1000.npz'),
    ('mega',        '../../Dataset/cvrp/distribution/cluster/2/cvrp_n100_1000.npz'),
    ('imbalanced',  '../../Dataset/cvrp/distribution/cluster/3/cvrp_n100_1000.npz'),
    ('overlap',     '../../Dataset/cvrp/distribution/cluster/4/cvrp_n100_1000.npz'),
    ('elongated',   '../../Dataset/cvrp/distribution/cluster/5/cvrp_n100_1000.npz'),
    ('bridge',      '../../Dataset/cvrp/distribution/cluster/6/cvrp_n100_1000.npz'),
    # annulus
    ('filled',  '../../Dataset/cvrp/distribution/annulus/area_uniform/cvrp_n100_1000.npz'),
    ('azimuth', '../../Dataset/cvrp/distribution/annulus/biased_ring/azimuth/cvrp_n100_1000.npz'),
    ('inward',  '../../Dataset/cvrp/distribution/annulus/biased_ring/in/cvrp_n100_1000.npz'),
    ('outward', '../../Dataset/cvrp/distribution/annulus/biased_ring/out/cvrp_n100_1000.npz'),
    ('thick',   '../../Dataset/cvrp/distribution/annulus/thick_ring/cvrp_n100_1000.npz'),
    ('thin',    '../../Dataset/cvrp/distribution/annulus/thin_ring/cvrp_n100_1000.npz'),
    # capacity
    ('cap20',  '../../Dataset/cvrp/capacity/20/cvrp_n100_1000.npz'),
    ('cap120', '../../Dataset/cvrp/capacity/120/cvrp_n100_1000.npz'),
    ('cap80',  '../../Dataset/cvrp/capacity/80/cvrp_n100_1000.npz'),
    ('cap200', '../../Dataset/cvrp/capacity/200/cvrp_n100_1000.npz'),
    ('cap250', '../../Dataset/cvrp/capacity/250/cvrp_n100_1000.npz'),
    # expansion
    ('axial',   '../../Dataset/cvrp/distribution/cvrp_expansion/exp_e1_axial/cvrp_n100_128.npz'),
    ('log',     '../../Dataset/cvrp/distribution/cvrp_expansion/exp_e2_log/cvrp_n100_128.npz'),
    ('wedge',   '../../Dataset/cvrp/distribution/cvrp_expansion/exp_e3_wedge/cvrp_n100_128.npz'),
    ('pinch',   '../../Dataset/cvrp/distribution/cvrp_expansion/exp_e4_pinch/cvrp_n100_128.npz'),
    ('shear',   '../../Dataset/cvrp/distribution/cvrp_expansion/exp_e5_shear/cvrp_n100_128.npz'),
    ('squeeze', '../../Dataset/cvrp/distribution/cvrp_expansion/exp_e6_squeeze/cvrp_n100_128.npz'),
    # implosion
    ('icircle',   '../../Dataset/cvrp/distribution/cvrp_implosion/impl_i1_circle/cvrp_n100_128.npz'),
    ('istar',     '../../Dataset/cvrp/distribution/cvrp_implosion/impl_i2_star/cvrp_n100_128.npz'),
    ('imulti',    '../../Dataset/cvrp/distribution/cvrp_implosion/impl_i3_multi/cvrp_n100_128.npz'),
    ('iline',     '../../Dataset/cvrp/distribution/cvrp_implosion/impl_i4_line/cvrp_n100_128.npz'),
    ('iellipse',  '../../Dataset/cvrp/distribution/cvrp_implosion/impl_i5_ellipse/cvrp_n100_128.npz'),
    ('igradient', '../../Dataset/cvrp/distribution/cvrp_implosion/impl_i6_gradient/cvrp_n100_128.npz'),
    # explosion
    ('xcircle',   '../../Dataset/cvrp/distribution/cvrp_explosion/expl_x1_circle/cvrp_n100_128.npz'),
    ('xstar',     '../../Dataset/cvrp/distribution/cvrp_explosion/expl_x2_star/cvrp_n100_128.npz'),
    ('xmulti',    '../../Dataset/cvrp/distribution/cvrp_explosion/expl_x3_multi/cvrp_n100_128.npz'),
    ('xring',     '../../Dataset/cvrp/distribution/cvrp_explosion/expl_x4_ring/cvrp_n100_128.npz'),
    ('xellipse',  '../../Dataset/cvrp/distribution/cvrp_explosion/expl_x5_ellipse/cvrp_n100_128.npz'),
    ('xgradient', '../../Dataset/cvrp/distribution/cvrp_explosion/expl_x6_gradient/cvrp_n100_128.npz'),
]


# path substring -> family name; first match wins
FAMILY_PATTERNS = [
    ("/in_distribution/", "in_dist"),
    ("/distribution/demand/", "demand"),
    ("/distribution/beta/", "beta"),
    ("/distribution/gaussian/", "gaussian"),
    ("/distribution/grid/", "grid"),
    ("/distribution/stripes/", "stripes"),
    ("/distribution/cluster/", "cluster"),
    ("/distribution/annulus/", "annulus"),
    ("/distribution/cvrp_expansion/", "expansion"),
    ("/distribution/cvrp_implosion/", "implosion"),
    ("/distribution/cvrp_explosion/", "explosion"),
    ("/cvrp/capacity/", "capacity"),
]


def family_of(path_str: str) -> str:
    for sub, name in FAMILY_PATTERNS:
        if sub in path_str:
            return name
    return "other"


def avg_vehicles(npz_path: Path) -> float:
    """Average vehicle count per instance.
    Prefers _mixveh.npy (actual model output) if it exists,
    otherwise falls back to ceil(sum_demand / capacity).
    """
    mixveh_path = npz_path.with_name(npz_path.stem + '_mixveh.npy')
    if mixveh_path.exists():
        return float(np.load(str(mixveh_path)).mean())
    data = np.load(str(npz_path), allow_pickle=True)
    raw = data['demands']
    demands    = (np.stack(raw) if raw.dtype == object else raw).astype(np.float32)
    capacities = data['capacities'].astype(np.float32)
    if capacities.ndim == 0:
        capacities = np.full(len(demands), float(capacities))
    return float(np.ceil(demands.sum(axis=1) / capacities).mean())


def load_lkh_avg_cost(npz_path: Path) -> float:
    """Paper reference: mean LKH-3 cost over the *feasible* instances.

    Reads the per-instance LKH outputs in the `*out*` dir next to the npz
    (CVRP_N100_I{i:05d}[_D{k}].lkhout.txt, i = 1..128), parses the final
    `Cost = <penalty>_<length>` line, drops instances with penalty > 0
    (LKH found no feasible solution, no .tour written) and returns
    mean(length) / 10000, in the same normalized space as POMO.
    """
    import re, glob
    d = npz_path.parent
    refs = [s for s in sorted(glob.glob(str(d / '*out*'))) if os.path.isdir(s)]
    if not refs:
        return None
    ref = refs[0]
    suffix = f"_D{d.parent.name}" if d.parent.parent.name == 'demand' else ''
    costs, n_infeas = [], 0
    for i in range(1, 129):
        p = os.path.join(ref, f"CVRP_N100_I{i:05d}{suffix}.lkhout.txt")
        if not os.path.exists(p):
            continue
        m = re.findall(r'Cost = (\d+)_(\d+)', open(p).read())
        if not m:
            continue
        pen, length = map(int, m[-1])
        if pen > 0:
            n_infeas += 1
            continue
        costs.append(length)
    if not costs:
        return None
    if n_infeas:
        print(f"    [LKH ref] {npz_path.parent.name}: {n_infeas} infeasible LKH instance(s) excluded, "
              f"{len(costs)} feasible used")
    return float(np.mean(costs)) / 10000.0


def _fmt_time(sec: float) -> str:
    if sec < 60:
        return f"{sec:.2f}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{int(m)}m{s:.1f}s"
    h, m = divmod(int(m), 60)
    return f"{h}h{int(m)}m{s:.1f}s"


def main():
    failures = []
    family_time  = defaultdict(float)
    family_count = defaultdict(int)
    family_gaps  = defaultdict(list)   # gap(%) per dataset in each family
    overall_t0   = time.perf_counter()
    per_run_records = []  # (name, fam, elapsed, mix_cost, lkh_cost, gap, nveh)

    for entry in NPZ_PATHS:
        if isinstance(entry, tuple):
            name, raw = entry
        else:
            raw, name = entry, os.path.basename(os.path.dirname(entry))
        npz_path = Path(raw)
        fam = family_of(str(npz_path))

        if not npz_path.exists():
            print(f"[SKIP] NPZ not found: {name} -> {npz_path}")
            failures.append((name, str(npz_path), "not found"))
            continue
        if npz_path.suffix.lower() != ".npz":
            print(f"[SKIP] Not an .npz file: {name} -> {npz_path}")
            failures.append((name, str(npz_path), "not .npz"))
            continue

        nveh     = avg_vehicles(npz_path)
        lkh_cost = load_lkh_avg_cost(npz_path)  # normalized (divided by 10000)
        lkh_str  = f"{lkh_cost:.4f}" if lkh_cost else "N/A"
        print(f"\n========== [{name}]  family={fam}  avg_veh={nveh:.1f}  lkh={lkh_str} ==========")

        t0       = time.perf_counter()
        mix_cost = None
        try:
            mix_cost = run_one(npz_path)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            failures.append((name, str(npz_path), repr(e)))
        elapsed = time.perf_counter() - t0

        gap = None
        if mix_cost is not None and lkh_cost is not None and lkh_cost > 0:
            gap = (mix_cost - lkh_cost) / lkh_cost * 100.0

        family_time[fam]  += elapsed
        family_count[fam] += 1
        if gap is not None:
            family_gaps[fam].append(gap)

        per_run_records.append((name, fam, elapsed, mix_cost, lkh_cost, gap, nveh))

        gap_str = f"  gap={gap:+.2f}%" if gap is not None else ""
        mix_str = f"  mix={mix_cost:.4f}" if mix_cost is not None else ""
        print(f"[DONE] [{name:12s}] family={fam:10s} veh={nveh:.0f}"
              f"{mix_str}  lkh={lkh_str}{gap_str}  time={_fmt_time(elapsed)}")

    overall = time.perf_counter() - overall_t0

    # families in NPZ_PATHS order
    seen, ordered = set(), []
    for entry in NPZ_PATHS:
        raw = entry[1] if isinstance(entry, tuple) else entry
        f = family_of(raw)
        if f not in seen:
            seen.add(f)
            ordered.append(f)

    print("\n========== Per-run summary ==========")
    print(f"  {'name':12s}  {'family':10s}  {'veh':>5s}  {'mix':>8s}  {'lkh':>8s}  {'gap':>8s}  {'time':>8s}")
    for name, fam, t, mix_cost, lkh_cost, gap, nveh in per_run_records:
        mix_s = f"{mix_cost:.4f}" if mix_cost  is not None else "  N/A  "
        lkh_s = f"{lkh_cost:.4f}" if lkh_cost  is not None else "  N/A  "
        gap_s = f"{gap:+.2f}%"    if gap       is not None else "  N/A  "
        print(f"  [{name:12s}] {fam:10s}  {nveh:5.1f}  {mix_s:>8s}  {lkh_s:>8s}  {gap_s:>8s}  {_fmt_time(t):>8s}")

    # ---- Paper Table 4 (CVRP row) ----
    # family gap = sum(model mean) / sum(LKH mean) - 1 over the family's variants (pooled);
    # the Demand column excludes the 'uniform' demand variant;
    # Avg = arithmetic mean of the 12 family columns (incl. in-dist Uniform).
    print("\n========== Paper Table 4 (CVRP): pooled gap per family ==========")
    paper_cols = []
    for fam in ordered:
        recs = [r for r in per_run_records
                if r[1] == fam and r[3] is not None and r[4] is not None
                and not (fam == "demand" and r[0] == "uniform")]
        if not recs:
            print(f"  {fam:12s}  N/A")
            continue
        g = (sum(r[3] for r in recs) / sum(r[4] for r in recs) - 1.0) * 100.0
        paper_cols.append(g)
        print(f"  {fam:12s}  gap={g:+.2f}%  (n_variants={len(recs)})")
    if paper_cols:
        print(f"  {'Avg':12s}  gap={float(np.mean(paper_cols)):+.2f}%  (mean of {len(paper_cols)} columns)")

    print("\n---------- (for reference) mean of per-variant gaps ----------")
    for fam in ordered:
        gaps = family_gaps.get(fam, [])
        if gaps:
            print(f"  {fam:12s}  avg_gap={float(np.mean(gaps)):+.2f}%  (n={len(gaps)})")
    print(f"\n========== Overall total: {_fmt_time(overall)} "
          f"({sum(family_count.values())} runs) ==========")

    if failures:
        print("\n=== Summary: failed / skipped ===")
        for nm, p, why in failures:
            print(f"  - [{nm}] {p}  ({why})")
        sys.exit(1)


if __name__ == "__main__":
    main()

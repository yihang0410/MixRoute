"""
CVRPTW distribution-shift test: one model x all 62 variants.

Test sets:
  - TW shift          8 (incl. the in-distribution `uniform` baseline)
  - Coord shift x 9  54 (annulus/beta/cluster/expansion/explosion/
                         gaussian/grid/implosion/stripes)

Data (under DATA_ROOT):
    cvrptw/<variant>/cvrptw_n100_{N}.npz                 <- TW shift
    cvrptw_<family>/<sub_variant>/cvrptw_n100_{N}.npz    <- coord shift

Output:
  - per variant: model no_aug / aug / gap / time
  - per-category table (13 categories: baseline + 3 TW + 9 coord) and the paper Table 4 block
  - overall total time + grand mean gap
  - CSV: test_ood_results_<model_name>.csv
"""

import csv
import os
import sys
import time
import importlib.util
from collections import defaultdict

import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

from CVRPTWEnv import CVRPTWEnv as Env
from CVRPTWProblemDef import augment_xy_data_by_8_fold


# Test config

MODEL = {
    'name':  'mixroute',
    'path':  '../../pretrained/cvrptw',
    'epoch': 6500,
    'src':   None,
}

DATA_ROOT = '../../Dataset/cvrptw/distribution'

POMO_REF_CATEGORIES = {'tw_timing'}
POMO_REF_FILE       = 'pomo_ref_costs.npy'

USE_CUDA        = True
CUDA_DEVICE_NUM = 0
NUM_SAMPLES     = 128
BATCH_SIZE      = 32
AUG_FACTOR      = 1

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

env_params = {
    'problem_size': 100,
    'pomo_size':    100,
}


# All variants (label, rel_path_under_DATA_ROOT, category)

# --- TW shift: 1 baseline + 3 width + 3 timing + 1 coupling = 8 ---
TW_GROUPS = {
    'tw_baseline':  ['uniform'],
    'tw_width':     ['tw_loose', 'tw_tight', 'tw_bimodal'],
    'tw_timing':    [ 'tw_late_rush', 'tw_narrow_band', 'tw_stripes_2'],
    'tw_coupling':  ['tw_decoupled'],
}

# --- Coord shift: 9 families × variants = 54 ---
COORD_GROUPS = {
    'coord_annulus':    ['area_uniform',
                         'biased_ring/azimuth', 'biased_ring/in', 'biased_ring/out',
                         'thick_ring', 'thin_ring'],
    'coord_beta':       ['0.1_0.1', '0.3_0.3', '0.3_1.7',
                         '0.6_1.4', '2.0_2.0', '4.0_4.0'],
    'coord_cluster':    ['1', '2', '3', '4', '5', '6'],
    'coord_expansion':  ['exp_e1_axial', 'exp_e2_log', 'exp_e3_wedge',
                         'exp_e4_pinch', 'exp_e5_shear', 'exp_e6_squeeze'],
    'coord_explosion':  ['expl_x1_circle', 'expl_x2_star', 'expl_x3_multi',
                         'expl_x4_ring', 'expl_x5_ellipse', 'expl_x6_gradient'],
    'coord_gaussian':   ['single/0.1', 'single/0.2', 'single/0.3', 'single/0.4'],
    'coord_grid':       ['grid', 'grid_aniso', 'grid_broken',
                         'grid_gauss', 'grid_rotate', 'grid_warp'],
    'coord_implosion':  ['impl_i1_circle', 'impl_i2_star', 'impl_i3_multi',
                         'impl_i4_line', 'impl_i5_ellipse', 'impl_i6_gradient'],
    'coord_stripes':    ['1', '2', '3', '4', '5', '6', '7', '8'],
}


# Switches for which variant groups to run
INCLUDE_TW_GROUPS    = True
INCLUDE_COORD_GROUPS = True


def _build_variant_list():
    """Returns ordered list of (label, npz_relpath_dir, category)."""
    items = []
    # TW
    if INCLUDE_TW_GROUPS:
        for cat, subs in TW_GROUPS.items():
            for sub in subs:
                label = sub
                items.append((label, f"cvrptw/{sub}", cat))
    # Coord
    if INCLUDE_COORD_GROUPS:
        for cat, subs in COORD_GROUPS.items():
            family_dir = cat.replace("coord_", "cvrptw_")  # e.g. coord_annulus -> cvrptw_annulus
            for sub in subs:
                # safe label: replace '/' with '_'
                label = f"{cat[len('coord_'):]}_{sub.replace('/', '_')}"
                items.append((label, f"{family_dir}/{sub}", cat))
    return items


VARIANTS = _build_variant_list()
# Short category names for the stdout tables (drop the 'coord_' / 'tw_' prefix)
CAT_LABEL = {
    'tw_baseline': 'baseline', 'tw_width': 'tw_width',
    'tw_timing':   'tw_timing', 'tw_coupling': 'tw_coupling',
    'coord_annulus':   'annulus',   'coord_beta':       'beta',
    'coord_cluster':   'cluster',   'coord_expansion':  'expansion',
    'coord_explosion': 'explosion', 'coord_gaussian':   'gaussian',
    'coord_grid':      'grid',      'coord_implosion':  'implosion',
    'coord_stripes':   'stripes',
}
CAT_ORDER = [
    'tw_baseline', 'tw_width', 'tw_timing', 'tw_coupling',
    'coord_annulus', 'coord_beta', 'coord_cluster',
    'coord_expansion', 'coord_explosion', 'coord_gaussian',
    'coord_grid', 'coord_implosion', 'coord_stripes',
]


# Helpers

def _fmt_time(sec):
    if sec < 60:
        return f"{sec:.2f}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{int(m)}m{s:.1f}s"
    h, m = divmod(int(m), 60)
    return f"{h}h{int(m)}m{s:.1f}s"


# Dynamic model loading

def load_model(src_dir, checkpoint_path, device):
    if src_dir is None:
        from CVRPTWModel import CVRPTWModel as Model
    else:
        spec = importlib.util.spec_from_file_location(
            "CVRPTWModel_custom", os.path.join(src_dir, "CVRPTWModel.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        Model = mod.CVRPTWModel

    model = Model(**model_params)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


# NPZ loading + LKH log parsing

def load_npz(npz_path, num_samples):
    data = np.load(npz_path, allow_pickle=True)
    coords   = data['coords']
    demands  = data['demands']
    tw       = data['time_windows']
    svc      = data['service_times']
    caps     = data['capacities']
    scale    = float(data['scale'])    if 'scale'    in data.files else 10000.0
    max_time = float(data['max_time']) if 'max_time' in data.files else 4.6

    B = min(len(coords), num_samples)
    N = coords[0].shape[0] - 1

    depot_xy     = np.zeros((B, 1, 2), dtype=np.float32)
    node_xy      = np.zeros((B, N, 2), dtype=np.float32)
    node_demand  = np.zeros((B, N),    dtype=np.float32)
    node_tw      = np.zeros((B, N, 2), dtype=np.float32)
    node_service = np.zeros((B, N),    dtype=np.float32)

    for i in range(B):
        c = np.asarray(coords[i]).astype(np.float32) / scale
        depot_xy[i, 0, :] = c[0]
        node_xy[i]        = c[1:]

        node_demand[i] = np.asarray(demands[i]).astype(np.float32) / float(caps[i])

        t = np.asarray(tw[i]).astype(np.float32) / scale
        node_tw[i] = t[1:]

        s = np.asarray(svc[i]).astype(np.float32) / scale
        node_service[i] = s[1:]

    return dict(
        depot_xy=torch.tensor(depot_xy),
        node_xy=torch.tensor(node_xy),
        node_demand=torch.tensor(node_demand),
        node_tw=torch.tensor(node_tw),
        node_service=torch.tensor(node_service),
        max_time=max_time,
        scale=scale,
    )


def parse_lkh_log(log_path):
    if not os.path.exists(log_path):
        return {}
    out = {}
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            try:
                idx     = int(parts[0])
                cost    = int(parts[6]) if parts[6] != 'NA' else None
                penalty = int(parts[7]) if parts[7] != 'NA' else None
                out[idx] = dict(cost=cost, penalty=penalty, status=parts[8])
            except (ValueError, IndexError):
                continue
    return out


# Env patch

def patch_env_load(env, depot_tw_end_aug):
    import types

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size
        depot_xy     = self.saved_depot_xy
        node_xy      = self.saved_node_xy
        node_demand  = self.saved_node_demand
        node_tw      = self.saved_node_tw
        node_service = self.saved_node_service

        if aug_factor == 8:
            self.batch_size = batch_size * 8
        elif aug_factor != 1:
            raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)

        depot_demand = torch.zeros(self.batch_size, 1, device=node_demand.device)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)

        depot_tw = torch.cat([
            torch.zeros(self.batch_size, 1, 1, device=node_tw.device),
            depot_tw_end_aug[:, :, None],
        ], dim=2)
        self.depot_node_tw = torch.cat((depot_tw, node_tw), dim=1)

        depot_service = torch.zeros(self.batch_size, 1, device=node_service.device)
        self.depot_node_service = torch.cat((depot_service, node_service), dim=1)

        self.dist_mat = (
            self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]
        ).norm(p=2, dim=-1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(
            self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(
            self.batch_size, self.pomo_size)

        self.reset_state.depot_xy     = depot_xy
        self.reset_state.node_xy      = node_xy
        self.reset_state.node_demand  = node_demand
        self.reset_state.node_tw      = node_tw
        self.reset_state.node_service = node_service

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX  = self.POMO_IDX

    env.load_problems = types.MethodType(load_problems, env)


# Model rollout

def run_pomo(model, env, data, batch_size, aug_factor, device):
    B = data['depot_xy'].shape[0]
    max_time = data['max_time']

    no_aug_list, aug_list = [], []

    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        bs = end - start

        depot_xy     = data['depot_xy'][start:end].to(device)
        node_xy      = data['node_xy'][start:end].to(device)
        node_demand  = data['node_demand'][start:end].to(device)
        node_tw      = data['node_tw'][start:end].to(device)
        node_service = data['node_service'][start:end].to(device)
        depot_tw_end = torch.full((bs, 1), max_time, device=device)

        if aug_factor == 8:
            depot_xy_a     = augment_xy_data_by_8_fold(depot_xy)
            node_xy_a      = augment_xy_data_by_8_fold(node_xy)
            node_demand_a  = node_demand.repeat(8, 1)
            node_tw_a      = node_tw.repeat(8, 1, 1)
            node_service_a = node_service.repeat(8, 1)
            depot_tw_end_a = depot_tw_end.repeat(8, 1)
        else:
            depot_xy_a, node_xy_a = depot_xy, node_xy
            node_demand_a, node_tw_a = node_demand, node_tw
            node_service_a, depot_tw_end_a = node_service, depot_tw_end

        env.FLAG__use_saved_problems = True
        env.saved_depot_xy     = depot_xy_a
        env.saved_node_xy      = node_xy_a
        env.saved_node_demand  = node_demand_a
        env.saved_node_tw      = node_tw_a
        env.saved_node_service = node_service_a
        env.saved_index        = 0
        patch_env_load(env, depot_tw_end_a)

        with torch.no_grad():
            env.load_problems(bs, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)

            state, reward, done = env.pre_step()
            while not done:
                selected, _ = model(state)
                state, reward, done = env.step(selected)

        aug_reward = reward.reshape(aug_factor, bs, env.pomo_size)
        max_pomo, _    = aug_reward.max(dim=2)
        no_aug         = -max_pomo[0, :]
        max_aug, _     = max_pomo.max(dim=0)
        aug            = -max_aug

        no_aug_list.append(no_aug.cpu().numpy())
        aug_list.append(aug.cpu().numpy())

    return np.concatenate(no_aug_list), np.concatenate(aug_list)


# Single-variant test

def test_one_variant(model, env, label, rel_path, category, device):
    var_dir = os.path.join(DATA_ROOT, rel_path)
    npz_path = os.path.join(var_dir, f"cvrptw_n100_{NUM_SAMPLES}.npz")
    log_path = os.path.join(var_dir, f"cvrptw_n100_{NUM_SAMPLES}.log")

    if not os.path.exists(npz_path):
        alt = os.path.join(var_dir, "cvrptw_n100_4.npz")
        if os.path.exists(alt):
            npz_path = alt
            log_path = alt.replace('.npz', '.log')
            print(f"    [INFO] {NUM_SAMPLES}-instance file not found, using cvrptw_n100_4.npz")
        else:
            return None

    data = load_npz(npz_path, NUM_SAMPLES)
    lkh  = parse_lkh_log(log_path)

    no_aug_scores, aug_scores = run_pomo(model, env, data, BATCH_SIZE, AUG_FACTOR, device)

    scale = data['scale']
    pomo_aug_scaled = aug_scores * scale

    if category in POMO_REF_CATEGORIES:
        ref_path = os.path.join(var_dir, POMO_REF_FILE)
        ref = np.load(ref_path) if os.path.exists(ref_path) else None
        gap = (float(no_aug_scores.mean()) / float(ref.mean()) - 1.0) * 100.0 if ref is not None else None
        if ref is None:
            print(f"    [WARN] file not found: {ref_path}")
        return {
            'variant': label, 'category': category, 'rel_path': rel_path,
            'samples': len(aug_scores),
            'pomo_no_aug': float(no_aug_scores.mean()), 'pomo_aug': float(aug_scores.mean()),
            'pomo_aug_int': float(pomo_aug_scaled.mean()),
            'lkh_feas': 0, 'lkh_mean': None,
            'mean_gap': gap, 'median_gap': None, 'min_gap': None, 'max_gap': None,
        }

    gaps, lkh_costs_feas = [], []
    for i in range(len(aug_scores)):
        e = lkh.get(i + 1)
        if e is None or e['cost'] is None or e['penalty']:
            continue
        c_lkh, c_pomo = e['cost'], pomo_aug_scaled[i]
        gaps.append((c_pomo - c_lkh) / c_lkh * 100.0)
        lkh_costs_feas.append(c_lkh)

    return {
        'variant':       label,
        'category':      category,
        'rel_path':      rel_path,
        'samples':       len(aug_scores),
        'pomo_no_aug':   float(no_aug_scores.mean()),
        'pomo_aug':      float(aug_scores.mean()),
        'pomo_aug_int':  float(pomo_aug_scaled.mean()),
        'lkh_feas':      len(lkh_costs_feas),
        'lkh_mean':      float(np.mean(lkh_costs_feas)) if lkh_costs_feas else None,
        'mean_gap':      float(np.mean(gaps)) if gaps else None,
        'median_gap':    float(np.median(gaps)) if gaps else None,
        'min_gap':       float(np.min(gaps)) if gaps else None,
        'max_gap':       float(np.max(gaps)) if gaps else None,
    }


# Main

def main():
    if USE_CUDA:
        torch.cuda.set_device(CUDA_DEVICE_NUM)
        device = torch.device('cuda', CUDA_DEVICE_NUM)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    print("=" * 80)
    print(f"MODEL:    {MODEL['name']}  @  epoch {MODEL['epoch']}")
    print(f"VARIANTS: {len(VARIANTS)} total "
          f"({sum(len(v) for v in TW_GROUPS.values())} TW shift + "
          f"{sum(len(v) for v in COORD_GROUPS.values())} coord shift)")
    print(f"AUG:      {AUG_FACTOR}x   batch={BATCH_SIZE}   samples per variant={NUM_SAMPLES}")
    print("=" * 80)

    ckpt = f"{MODEL['path']}/checkpoint-{MODEL['epoch']}.pt"
    model = load_model(MODEL['src'], ckpt, device)
    env = Env(**env_params)

    rows = []
    cat_time     = defaultdict(float)
    cat_count    = defaultdict(int)
    cat_no_aug   = defaultdict(list)
    cat_aug      = defaultdict(list)

    overall_t0 = time.perf_counter()

    for k, (label, rel_path, category) in enumerate(VARIANTS, start=1):
        cat_short = CAT_LABEL.get(category, category)
        print(f"\n[{k:3d}/{len(VARIANTS)}] {label:32s}  ({cat_short})")

        t0 = time.perf_counter()
        try:
            res = test_one_variant(model, env, label, rel_path, category, device)
        except Exception as e:
            print(f"    [ERROR] {type(e).__name__}: {e}")
            res = None
        elapsed = time.perf_counter() - t0

        if res is None:
            rows.append({
                'variant': label, 'category': category, 'rel_path': rel_path,
                'samples': 0,
                'pomo_no_aug': float('nan'), 'pomo_aug': float('nan'),
                'pomo_aug_int': float('nan'),
                'lkh_feas': 0, 'lkh_mean': None,
                'mean_gap': None, 'median_gap': None,
                'min_gap': None, 'max_gap': None,
                'time_sec': elapsed,
            })
            continue

        res['time_sec'] = elapsed
        rows.append(res)
        cat_time[category]  += elapsed
        cat_count[category] += 1
        cat_no_aug[category].append(res['pomo_no_aug'])
        cat_aug[category].append(res['pomo_aug'])

        gap_str = (f"{res['mean_gap']:+.2f}%" if res['mean_gap'] is not None else "NA")
        print(f"    -> model aug={res['pomo_aug']:.4f}  "
              f"gap={gap_str}  time={_fmt_time(elapsed)}")

    overall = time.perf_counter() - overall_t0

    # ---- per-variant table ----
    print("\n" + "=" * 116)
    print("PER-VARIANT SUMMARY")
    print("=" * 116)
    header = (f"{'variant':<32s} {'cat':>10s} {'#samp':>5s} "
              f"{'no_aug':>10s} {'aug':>10s} {'lkh':>10s} "
              f"{'mean_gap%':>10s} {'med_gap%':>10s} {'min%':>8s} {'max%':>8s} {'time':>10s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        cat_short = CAT_LABEL.get(r['category'], r['category'])
        lkh_s = f"{r['lkh_mean']:.0f}" if r.get('lkh_mean') is not None else "NA"
        mg_s  = f"{r['mean_gap']:+.2f}"  if r['mean_gap']  is not None else "NA"
        medg  = f"{r['median_gap']:+.2f}" if r['median_gap'] is not None else "NA"
        ming  = f"{r['min_gap']:+.2f}"   if r['min_gap']   is not None else "NA"
        maxg  = f"{r['max_gap']:+.2f}"   if r['max_gap']   is not None else "NA"
        print(f"{r['variant']:<32s} {cat_short:>10s} {r['samples']:>5d} "
              f"{r['pomo_no_aug']:>10.4f} {r['pomo_aug']:>10.4f} {lkh_s:>10s} "
              f"{mg_s:>10s} {medg:>10s} {ming:>8s} {maxg:>8s} {_fmt_time(r['time_sec']):>10s}")

    # ---- per-category aggregate (sample-weighted mean gap) ----
    print("\n" + "=" * 90)
    print("PER-CATEGORY SUMMARY")
    print("=" * 90)
    cat_gaps     = defaultdict(list)
    cat_samples  = defaultdict(int)
    for r in rows:
        if r['mean_gap'] is None:
            continue
        cat_gaps[r['category']].append((r['mean_gap'], r['samples']))
        cat_samples[r['category']] += r['samples']

    print(f"{'category':<14s} {'#var':>5s} {'#samp':>7s} "
          f"{'no_aug':>10s} {'aug':>10s} {'mean_gap%':>10s} {'median%':>10s} {'total_time':>14s}")
    for cat in CAT_ORDER:
        if cat_count[cat] == 0:
            continue
        gaps = cat_gaps[cat]
        if gaps:
            tot_w = sum(s for _, s in gaps)
            mg = sum(g * s for g, s in gaps) / tot_w if tot_w else float('nan')
            mg_s = f"{mg:+.2f}"
            med_s = f"{np.median([g for g, _ in gaps]):+.2f}"
        else:
            mg_s = "NA"; med_s = "NA"
        n_var  = cat_count[cat]
        n_sam  = cat_samples[cat]
        t_tot  = cat_time[cat]
        na_avg = sum(cat_no_aug[cat]) / len(cat_no_aug[cat]) if cat_no_aug[cat] else float('nan')
        ag_avg = sum(cat_aug[cat])    / len(cat_aug[cat])    if cat_aug[cat]    else float('nan')
        print(f"{CAT_LABEL.get(cat, cat):<14s} {n_var:>5d} {n_sam:>7d} "
              f"{na_avg:>10.4f} {ag_avg:>10.4f} {mg_s:>10s} {med_s:>10s} {_fmt_time(t_tot):>14s}")

    # ---- paper Table 4 (CVRPTW row): one column per category, Avg = arithmetic mean ----
    print("\n" + "=" * 90)
    print("PAPER TABLE 4 (CVRPTW): gap per category")
    print("=" * 90)
    paper_cols = []
    for cat in CAT_ORDER:
        gaps = cat_gaps.get(cat)
        if not gaps:
            continue
        tot_w = sum(s for _, s in gaps)
        g = sum(x * s for x, s in gaps) / tot_w
        paper_cols.append(g)
        print(f"  {CAT_LABEL.get(cat, cat):<14s} gap={g:+.2f}%  (n_variants={len(gaps)})")
    if paper_cols:
        print(f"  {'Avg':<14s} gap={float(np.mean(paper_cols)):+.2f}%  (mean of {len(paper_cols)} columns)")

    # ---- overall ----
    overall_gaps = [(r['mean_gap'], r['samples']) for r in rows
                    if r['mean_gap'] is not None]
    if overall_gaps:
        tot_w = sum(s for _, s in overall_gaps)
        overall_mean_gap = sum(g * s for g, s in overall_gaps) / tot_w if tot_w else float('nan')
    else:
        overall_mean_gap = float('nan')

    total_samples = sum(r['samples'] for r in rows)
    all_no_aug = [r['pomo_no_aug'] for r in rows if not np.isnan(r['pomo_no_aug'])]
    all_aug    = [r['pomo_aug']    for r in rows if not np.isnan(r['pomo_aug'])]
    overall_no_aug = sum(all_no_aug) / len(all_no_aug) if all_no_aug else float('nan')
    overall_aug    = sum(all_aug)    / len(all_aug)    if all_aug    else float('nan')
    print("\n" + "=" * 90)
    print(f"OVERALL: {len(rows)} variants, {total_samples} instances, "
          f"no_aug={overall_no_aug:.4f}, aug={overall_aug:.4f}, "
          f"mean_gap={overall_mean_gap:+.2f}%, total_time={_fmt_time(overall)}")
    print("=" * 90)

    # ---- CSV ----
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"test_ood_results_{MODEL['name']}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            'variant', 'category', 'rel_path', 'samples',
            'pomo_no_aug', 'pomo_aug', 'pomo_aug_int',
            'lkh_feas', 'lkh_mean',
            'mean_gap', 'median_gap', 'min_gap', 'max_gap',
            'time_sec',
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    main()

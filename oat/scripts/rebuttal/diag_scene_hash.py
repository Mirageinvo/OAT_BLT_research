"""Where do scene-hash mismatches come from? (no sim; reads episodes.csv of the P0 main block)

  .venv/bin/python scripts/rebuttal/diag_scene_hash.py eval_out/rebuttal_cs_ablation/main/{base,cs_prefix,cs_endpoint,medoid_prefix,cs_prefix_fixed}

Per (task, init_id) collects the hash of every (variant, repeat_seed) run and reports:
  - how many distinct hashes a scene has;
  - whether a mismatching scene's hash is a function of repeat_seed only / variant only / neither;
  - mismatch rate by batch position (init_id % n_envs), batch index, task, variant, seed;
  - whether deviating runs are explained by the PREVIOUS episode in the same worker slot
    (same variant+seed, init_id - n_envs): its success / env_steps.
"""
import sys
from collections import Counter

import pandas as pd

dirs = sys.argv[1:]
ep = pd.concat([pd.read_csv(f"{d}/episodes.csv") for d in dirs], ignore_index=True)
n_envs = int(ep['batch_envs'].max())
ep['slot'] = ep['init_id'] % n_envs
ep['batch'] = ep['init_id'] // n_envs
print(f"rows={len(ep)} variants={sorted(ep.variant.unique())} seeds={sorted(ep.repeat_seed.unique())} n_envs={n_envs}")

g = ep.groupby(['task', 'init_id'])
nd = g['scene_hash'].nunique()
print("\n# distinct hashes per scene (over all variant x seed runs):", dict(Counter(nd.values)))
bad = nd[nd > 1].index
ep['scene_bad'] = ep.set_index(['task', 'init_id']).index.isin(bad)

# majority hash per scene; a run "deviates" if its hash != majority
maj = g['scene_hash'].agg(lambda s: s.value_counts().idxmax()).rename('maj')
ep = ep.join(maj, on=['task', 'init_id'])
ep['dev'] = ep['scene_hash'] != ep['maj']
print(f"deviating runs: {int(ep.dev.sum())} / {len(ep)}")

kinds = Counter()
for (t, i), s in ep[ep.scene_bad].groupby(['task', 'init_id']):
    by_seed = s.groupby('repeat_seed')['scene_hash'].nunique().max() == 1
    by_var = s.groupby('variant')['scene_hash'].nunique().max() == 1
    kinds['function of seed only' if by_seed and not by_var else
          'function of variant only' if by_var and not by_seed else
          'neither (run-specific)' if not (by_seed or by_var) else 'both?'] += 1
print("mismatching scenes, hash structure:", dict(kinds))

for col in ['slot', 'variant', 'repeat_seed', 'task']:
    print(f"\n# deviation rate by {col}")
    print(ep.groupby(col)['dev'].agg(['mean', 'sum', 'count']).round(3).to_string())
print("\n# deviation rate by batch index (first 12 and overall first batch)")
print(ep.groupby('batch')['dev'].mean().round(3).head(12).to_string())

# previous episode in the same worker slot (same variant, seed): init_id - n_envs
prev = ep[['variant', 'repeat_seed', 'init_id', 'success', 'env_steps']].copy()
prev['init_id'] = prev['init_id'] + n_envs
prev = prev.rename(columns={'success': 'prev_success', 'env_steps': 'prev_steps'})
ep = ep.merge(prev, on=['variant', 'repeat_seed', 'init_id'], how='left')
print("\n# deviation vs previous episode in the same slot (NaN = first batch of the seed)")
ep['prev_kind'] = ep['prev_success'].map({1: 'prev success', 0: 'prev failure'}).fillna('first batch')
print(ep.groupby('prev_kind')['dev'].agg(['mean', 'sum', 'count']).round(3).to_string())
ep['prev_len'] = pd.cut(ep['prev_steps'], [0, 200, 300, 400, 549, 10000])
print(ep.groupby('prev_len', observed=True)['dev'].agg(['mean', 'sum', 'count']).round(3).to_string())

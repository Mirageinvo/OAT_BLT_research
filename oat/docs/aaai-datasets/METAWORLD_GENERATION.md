# MetaWorld demo generation — technical notes (AAAI datasets)

This file documents **how our MetaWorld Zarr differs** from downloading pre-built demos elsewhere.

## Generator

- Script: `scripts/gen_metaworld_data.py`
- Cluster log (single-task regen): `logs/metaworld_single_data_regen.log`
- Multitask log: `logs/gen_metaworld_mt4_N50.log`

## Expert

- MetaWorld v2 **oracle / expert policies** roll out until horizon or success.
- **4× RGB** 128×128: `corner`, `corner2`, `corner3`, `behindGripper`
- **`agent_pos`**: 9D
- **Action**: 4D

## Episode filter (critical)

```python
episode_success_count += int(bool(info.get("success", False)))
if episode_success_count == 0:
    attempt_idx += 1
    continue  # reject — do not advance episode_idx
```

→ Episode is **accepted if success is True on any step**, not only the last step.

We briefly tested **terminal-only** success; on successful rollouts it was equivalent, but we kept the any-step rule to match `mt4_N50` and avoid rejecting valid expert demos.

## Reset seed (critical fix, 2026-07-10)

Without `seed`, our `MetaworldEnv.reset()` restored the **same** MuJoCo init snapshot → infinite reject loop on hard inits.

```python
roll_seed = episode_idx * 1_000_000 + attempt_idx
obs_dict, _ = env.reset(seed=roll_seed)
```

## Relation to sim-env

Env/runner ported from [Chaoqi-LIU/sim-env](https://github.com/Chaoqi-LIU/sim-env).  
We do **not** ship sim-env's original demo files; we **regenerated** 50 eps/task under our port.

**Paper stance:** MetaWorld results are interpreted **within this demo port** (see paper limitations).

## Single-task vs MT4

| File | Use in paper |
|------|----------------|
| `{task}_N50.zarr` | **Table P** (single-task specialist training) |
| `mt4_N50.zarr` | Exploratory multitask only |

Split from MT4 (`scripts/split_metaworld_mt4_zarr.py`, round-robin `episode i → task i%4`) is byte-identical to source episodes for those indices.

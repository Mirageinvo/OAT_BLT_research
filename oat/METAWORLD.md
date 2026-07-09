# MetaWorld (MT4) — OAT pipeline

Reproduce the OAT paper benchmark on **MetaWorld MT4** with the same stack as LIBERO / RoboMimic:

**data → tokenizer → policy → BoN → AWR**

## Paper protocol (Appendix A)

| Item | Value |
|------|-------|
| Suite | **MT4**: `box-close`, `coffee-pull`, `disassemble`, `stick-pull` |
| Demos | **50** successful expert trajectories **per task** (200 total for MT4) |
| Action dim | **4** |
| Obs | 4× RGB 128×128 (`corner`, `corner2`, `corner3`, `behindGripper`) + `agent_pos` (9D) |
| Eval | **250** rollouts (`n_test=250`, 5 experiment repeats via `--num_exp 5`) |
| Reference SR | OAT8 ≈ **24.4%** mean success |

## Branch & layout

```text
git checkout metaworld   # branched from robomimic

oat/oat/env/metaworld/           # MetaworldEnv + factory (from sim-env)
oat/oat/env_runner/metaworld_runner.py
oat/oat/config/task/{policy,tokenizer}/metaworld/mt4.yaml
scripts/gen_metaworld_data.py
scripts/prepare_metaworld_mt4.sh
scripts/eval_metaworld_{policy,bon,awr}.sh
slurm/metaworld/
```

## One-time setup (cluster / docker)

MetaWorld v2 needs **MuJoCo 2.1.0** (not MuJoCo 3 / `mujoco` pip used by robosuite 1.4):

```bash
# ~/.bashrc (or docker entrypoint)
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

cd oat
uv sync   # installs metaworld from rlworkgroup/metaworld (see pyproject.toml)
bash scripts/prepare_metaworld_mt4.sh setup_mujoco
```

**Note:** MetaWorld and RoboMimic/LIBERO use different MuJoCo stacks. On a shared node, prefer a **dedicated docker/venv** for MetaWorld runs (same pattern as robomimic docker).

## Pipeline commands

```bash
cd oat

# 1) Expert demos → Zarr (~30–90 min on GPU, rejects failed expert rollouts)
#    N50 means 50 demos per task = 200 MT4 episodes total.
MUJOCO_GL=egl bash scripts/prepare_metaworld_mt4.sh gen_data
# → data/metaworld/mt4_N50.zarr

# 2) Tokenizer (top-3 by test_reconst_mse)
bash scripts/prepare_metaworld_mt4.sh tok
export TOKENIZER_CKPT=output/.../ep-xxxx_mse-0.00x.ckpt

# 3) Policy (sim eval every 200 ep by default; top-3 by mean_success_rate)
bash scripts/prepare_metaworld_mt4.sh policy
export POLICY_CKPT=output/.../ep-xxxx_sr-0.2xx.ckpt

# 4) Eval baseline (OAT8, single sample)
bash scripts/prepare_metaworld_mt4.sh eval_base

# 5) BoN (verifier-free vote, N=8)
BON_N=8 bash scripts/prepare_metaworld_mt4.sh eval_bon

# 6) AWR BoN-distillation
bash scripts/prepare_metaworld_mt4.sh awr_collect
bash scripts/prepare_metaworld_mt4.sh awr_train
bash scripts/prepare_metaworld_mt4.sh eval_awr
```

### Hydra overrides (manual)

```bash
# Tokenizer
accelerate launch scripts/run_workspace.py --config-name=train_oattok \
  task/tokenizer=metaworld/mt4 training.num_demo=50

# Policy
accelerate launch scripts/run_workspace.py --config-name=train_oatpolicy \
  task/policy=metaworld/mt4 \
  policy.action_tokenizer.checkpoint="${TOKENIZER_CKPT}" \
  training.num_demo=50 training.rollout_every=200
```

## SLURM (MIPT cluster)

```bash
cd oat
sbatch slurm/metaworld/gen_data_mt4.slurm
sbatch slurm/metaworld/train_tok_mt4.slurm
# after tok ckpt:
TOKENIZER_CKPT=... sbatch slurm/metaworld/train_policy_mt4.slurm
```

## Differences vs RoboMimic port

| | RoboMimic | MetaWorld MT4 |
|---|-----------|---------------|
| Data source | HDF5 mh demos | On-the-fly expert (`gen_metaworld_data.py`) |
| `num_demo` | 200 | **50** |
| Action dim | 7 | **4** |
| State | eef pose + quat + gripper | `agent_pos` (9D) |
| Cameras | 2 @ 84² | **4 @ 128²** |
| Eval `n_test` | 50 | **250** |
| MuJoCo | mujoco_py / robosuite 1.4 | **mujoco210** + metaworld v2 |

## Status / TODO

- [x] Env + runner + configs ported from [sim-env](https://github.com/Chaoqi-LIU/sim-env)
- [x] `collect_awr_dataset.py` + `branch_value_k.py` benchmark wiring
- [x] `prepare_metaworld_mt4.sh` end-to-end script
- [ ] Run `gen_data` on cluster (validate zarr shapes)
- [ ] Train tok + policy; compare SR to paper ~24.4%
- [ ] BoN + AWR replication (LIBERO-style diagnosis on 2nd benchmark)

## Troubleshooting

- **`ImportError: metaworld`**: `uv sync` in `oat/`
- **`GLFW` / render errors**: set `MUJOCO_GL=egl`, ensure GPU visible in docker
- **Expert gen stuck**: expert rejects episodes with `<5` consecutive successes; increase wall time or lower `--num_episodes` for smoke (`--num_episodes 8` must be divisible by 4)
- **`num_episodes` not divisible**: MT4 requires `num_episodes % 4 == 0`

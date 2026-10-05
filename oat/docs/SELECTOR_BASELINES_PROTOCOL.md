# Selector Baselines Protocol (paper revision)

## Scope
RoboMimic + MetaWorld + RoboCasa only (matched Table P). Not LIBERO.

## Repo / branch
- Code branch: `aamas27_selector_baselines`
- Artifact HF: `hackhackhack66666/aaai27-models`
- Cluster: **aicenter4** (`lab-c-0`). SSH: [`AICENTER4_ACCESS.md`](AICENTER4_ACCESS.md) (`ssh aicenter4-jump`). Latency GPU must be empty.

## Matched eval (canonical)
```
--use_k_tokens 8 --entropy_threshold 0
--temperature 1.0 --topk 10
--bon_free 8 --bon_signal {vote|random|medoid|max_likelihood}
--n_test 50 --num_exp 5 --test_start_seed 10000
--n_parallel_envs 8   # maximize GPU2 throughput
--selector_seed 0     # for random only
```
Rollouts/method/suite: 50 × 5 = **250**.

## Base checkpoints (Wave1 OAT8 — NOT AWR)
| Suite | BASE_CKPT |
|-------|-----------|
| can | `.../can_N200/checkpoints/ep-1700_sr-0.940.ckpt` |
| lift (paper B) | `.../lift_N200/checkpoints/ep-1400_sr-0.950.ckpt` |
| square | `.../square_N200/checkpoints/ep-0700_sr-0.420.ckpt` |
| coffee-pull | `.../ep-1000_sr-0.432.ckpt` |
| stick-pull | `.../ep-0800_sr-0.212.ckpt` |
| disassemble | `.../ep-1400_sr-0.700.ckpt` |
| box-close | `.../ep-2000_sr-0.552.ckpt` |
| RC close_drawer | `my_models/robocasa_close_drawer_topk_ep0500_sr0.700.ckpt` |
| RC coffee | `my_models/robocasa_coffee_press_button_topk_ep0500_sr0.600.ckpt` |
| RC sink | `my_models/robocasa_turn_off_sink_faucet_topk_ep0500_sr0.580.ckpt` |
| RC microwave | `my_models/robocasa_turn_off_microwave_topk_ep0500_sr0.620.ckpt` |

Paper anchors (do not replace): CS/vote N=8 numbers from `RESULTS.md` Table P.

## Selectors
| signal | rule |
|--------|------|
| vote | KDE density on normalized executed prefix (CS) |
| medoid | argmin sum L2 to others on same prefix |
| max_likelihood | argmax sum log p(z_k \| …) under T=1 + topk=10 |
| random | Uniform index via isolated CPU Generator |

Shared: one vision encode, N=8 AR samples, same detokenize. Only selector index differs.

## Priority order (GPU time)
1. Smoke Can (10 rollouts) all 4 signals
2. Full Can: random → medoid → max_likelihood (vote already known)
3. Stick-pull, box-close (large CS Δ)
4. Remaining RM/MW
5. RoboCasa
6. Latency

## Outputs
`output/eval/aamas27_selector_baselines/<suite>/<signal>_n8/`

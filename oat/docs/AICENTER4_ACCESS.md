# Как зайти на aicenter4 (lab-c-0)

Канонический хост для paper eval / Table 5 latency. **3× H100**. Home: `/workspace/home/askhabaliev_gs`.

`ccmplanner` — **только hop**, не нода. Не называть его aic4.

---

## 1. Всегда так (канон)

С ноута, ключ `~/.ssh/mipt_lab`, юзер **`askhabaliev_gs`** (со **s**):

```bash
ssh aicenter4-jump
```

Это в `~/.ssh/config`:

```
Host ccmplanner
  HostName 100.98.148.137
  User askhabaliev_gs
  IdentityFile ~/.ssh/mipt_lab
  IdentitiesOnly yes

Host aicenter4-jump
  HostName 10.55.230.26
  Port 30025
  User askhabaliev_gs
  IdentityFile ~/.ssh/mipt_lab
  IdentitiesOnly yes
  ProxyCommand ssh -i ~/.ssh/mipt_lab -o IdentitiesOnly=yes askhabaliev_gs@100.98.148.137 -W %h:%p
```

Эквивалент без алиаса:

```bash
ssh -o ProxyCommand='ssh -i ~/.ssh/mipt_lab -o IdentitiesOnly=yes askhabaliev_gs@100.98.148.137 -W %h:%p' \
  -i ~/.ssh/mipt_lab -o IdentitiesOnly=yes -p 30025 askhabaliev_gs@10.55.230.26
```

Проверка, что это aic4: `hostname` → **`lab-c-0`**, `nvidia-smi -L` → три H100.

Ключ (класть в `authorized_keys` **этого** юзера):

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIO4XmfyULqQvbidLA5sFrJmkotZmDtORKvy5TSX5Vl61 askhabalievvv91@internet.ru
```

SHA256: `jfzgmmXDdclCff5xZrS1vyakNWfpF2Mcp6OmekWHFQg`

---

## 2. Что живое, что нет

| путь | статус | зачем |
|---|---|---|
| `ssh aicenter4-jump` → `10.55.230.26:30025` | **рабочий** | user-sshd, принимает `mipt_lab` |
| `10.55.230.26:22` | нода жива, **наш ключ не принимают** | хостовый sshd, другой набор ключей |
| `Host aicenter4` / NetBird `100.98.*` | часто мёртв / IP плывёт | не канон. Видели `52.170`, `193.248`, `184.195`, `248.90` |
| `ssh -J labjump@proxy2.cod.phystech.edu:10228 -p 30025 …` | **нам нет** | `labjump` не имеет нашего ключа (Permission denied на jump) |
| `askhabaliev_g` (без s) | **не наш аккаунт** | опечатка в письме Владу; ключ туда класть бесполезно |

PasswordAuthentication на aic4 выключен — только ключ.

---

## 3. Если `aicenter4-jump` не пускает

С **ccmplanner** (он обычно жив):

```bash
ssh ccmplanner
# порт нашего sshd
(echo >/dev/tcp/10.55.230.26/30025) >/dev/null 2>&1 && echo OPEN || echo CLOSED
(echo >/dev/tcp/10.55.230.26/22) >/dev/null 2>&1 && echo p22=OPEN || echo p22=CLOSED
# NetBird peer (IP плывёт)
netbird status --json | python3 -c 'import json,sys
d=json.load(sys.stdin)
for p in d["peers"]["details"]:
  if "aicenter4" in str(p).lower():
    print(p.get("fqdn"), p.get("netbirdIp"), p.get("status"))'
```

Читать так:

- **:22 OPEN, :30025 CLOSED** — железо живо, **наш sshd упал**. Раны на GPU могли выжить (это не reboot). Ждать, пока поднимут 30025, или root/консоль.
- **оба CLOSED** — нода/сеть легла.
- **:30025 OPEN, jump всё равно refused** — сломан hop ccmplanner, не aic4.
- NetBird `Connecting` / stale IP — игнорировать, идти на `10.55.230.26:30025`.

Не ждать `Host aicenter4` (100.98.52.170) — этот IP в конфиге **устарел**.

---

## 4. После входа — раны

Код:

| дерево | зачем |
|---|---|
| `~/oat_code_kdpe` | eval LIBERO / selectors (`cell_libero.sh`) |
| `~/oat_code_seed` | latency Table 5 (`LAST_CALL` в `kdpe.py`) |
| `~/cs_libero_hf/policy_ep-0250_sr-0.596.ckpt` | LIBERO ckpt |
| `~/aaai27_infra/models/checkpoints/selected_from_output/...` | Wave1 RM/MW ckpts |
| `~/gl-prefix/usr/lib/x86_64-linux-gnu` | user-space OSMesa (системного libOSMesa нет) |
| `~/queues/cell_libero.sh` | LIBERO cell |
| `~/queues/lane_latency_table5_gpu2.sh` | очередь latency GPU2 |
| `~/oat_eval_out/paired/libero10/<method>/eval_log.json` | paper LIBERO [E] |

Слоты (не убивать чужие/здоровые наши eval):

- считать **parent** `eval_policy_sim.py`, чей PPID — `cell_*.sh`. Воркеры AsyncVectorEnv наследуют cmdline и раздувают `pgrep`.
- max **2 eval на GPU** (parent). GPU2: ≤2 RoboCasa OSMesa **или** одна latency-очередь на **пустой** карте.
- HAMI часто показывает 0 MiB при живых процессах — смотреть `nvidia-smi` Processes + `CUDA_VISIBLE_DEVICES`.

LIBERO (latency GPU2 **закрыта**; vote/random уже есть — не `--force` их). На **3 GPU**:

```bash
export LD_LIBRARY_PATH=$HOME/gl-prefix/usr/lib/x86_64-linux-gnu
nohup env GPU=0 LD_LIBRARY_PATH=$LD_LIBRARY_PATH bash ~/queues/cell_libero.sh max_likelihood &
nohup env GPU=0 LD_LIBRARY_PATH=$LD_LIBRARY_PATH bash ~/queues/cell_libero.sh base &
nohup env GPU=1 LD_LIBRARY_PATH=$LD_LIBRARY_PATH bash ~/queues/cell_libero.sh kdpe &
nohup env GPU=2 LD_LIBRARY_PATH=$LD_LIBRARY_PATH bash ~/queues/cell_libero.sh medoid &
```

`cell_libero.sh` сам ставит `--force` на свой out. vote/random с готовым `eval_log.json` скрипт **скипает**.

Latency Table 5 (канон = can `selector_latency_gpu1_n32diag`): warmup=50, 8 trials × 20 reps, Ns=8,16,32, batch 1, реальные кандидаты. **Только пустая карта. По одному ckpt. can уже снят — не перегонять. RoboCasa KDPE не мерить (D=12).** Очередь: `bash ~/queues/lane_latency_table5_gpu2.sh` при `CUDA_VISIBLE_DEVICES=2` из `oat_code_seed`.

Не ждать токен в терминале. Не `pkill` по слишком широкому паттерну (убьёт свой bash).

---

## 5. Что писать Владу, если снова отрежет

1. Юзер **`askhabaliev_gs`**, путь ключа: `/workspace/home/askhabaliev_gs/.ssh/authorized_keys` (не `askhabaliev_g`).
2. Нужен **user-sshd на :30025** (не только хостовый :22).
3. Обход `-J labjump@proxy2.cod.phystech.edu:10228` заработает, только если тот же pubkey лежит у **`labjump`**.
4. Ничего не удалять на `/workspace` без явного «удаляй».

---

## 6. Не путать хосты

| хост | что это | наши раны paper? |
|---|---|---|
| **aicenter4 / lab-c-0** | 3× H100, `10.55.230.26:30025` | **да** |
| **ccmplanner** | 2× V100, hop `100.98.148.137` | нет (кроме старого RC, часть discarded) |
| **aicenter1** | 2× A100, hop/запас | нет, карты обычно чужие |

Последний живой вход этим путём: 2026-10-04 (после отвала 30025 он снова открылся; NetBird aic4 был `Connecting`).

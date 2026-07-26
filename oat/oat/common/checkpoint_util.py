from typing import Optional, Dict
import os
import re


class TopKCheckpointManager:
    """Keep at most ``k`` scored checkpoints on disk.

    On init (and therefore on every resume), rebuilds the in-memory map from
    existing ``*.ckpt`` files in ``save_dir`` and deletes orphans beyond top-k.
    ``latest.ckpt`` is never managed here (``save_last_ckpt`` overwrites it).
    """

    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
        self._rebuild_from_disk()

    def _parse_score(self, filename: str) -> Optional[float]:
        """Extract monitor score from a topk filename.

        Supports the repo's usual patterns, e.g.
        ``ep-0600_sr-0.280.ckpt``, ``ep-1940_mse-0.003.ckpt``.
        """
        # Prefer explicit tag matching monitor_key suffix (mean_success_rate → sr)
        key = self.monitor_key.lower()
        aliases = []
        if 'success' in key or key.endswith('sr') or key == 'sr':
            aliases.append('sr')
        if 'mse' in key or 'reconst' in key:
            aliases.append('mse')
        if 'loss' in key:
            aliases.append('loss')
        if 'score' in key:
            aliases.append('score')
        # Always try generic last-float tag: _xxx-0.123.ckpt
        aliases.extend(['sr', 'mse', 'loss', 'score'])

        seen = set()
        for tag in aliases:
            if tag in seen:
                continue
            seen.add(tag)
            m = re.search(rf'(?:^|_){re.escape(tag)}-([0-9]*\.?[0-9]+)\.ckpt$', filename)
            if m:
                return float(m.group(1))
        return None

    def _rebuild_from_disk(self) -> None:
        if self.k == 0 or not os.path.isdir(self.save_dir):
            self.path_value_map = dict()
            return

        entries = []
        for name in os.listdir(self.save_dir):
            if not name.endswith('.ckpt'):
                continue
            if name == 'latest.ckpt' or name.startswith('latest'):
                continue
            path = os.path.join(self.save_dir, name)
            if not os.path.isfile(path):
                continue
            value = self._parse_score(name)
            if value is None:
                continue
            entries.append((path, value))

        reverse = self.mode == 'max'
        entries.sort(key=lambda x: x[1], reverse=reverse)
        keep = entries[: self.k]
        for path, _ in entries[self.k :]:
            try:
                os.remove(path)
            except OSError:
                pass
        self.path_value_map = {p: v for p, v in keep}

    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None
        if self.monitor_key not in data:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))

        # Same path already tracked (re-eval same epoch) — just refresh score.
        if ckpt_path in self.path_value_map:
            self.path_value_map[ckpt_path] = value
            return ckpt_path

        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path

        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path

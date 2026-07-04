from typing import List

SUPPORTED_TASKS = ("lift", "can", "square")


def get_subtasks(task_name: str) -> List[str]:
    """Return rollout subtasks for a RoboMimic benchmark name."""
    task_key = task_name.lower()
    if task_key in SUPPORTED_TASKS:
        return [task_key]
    raise ValueError(f"Unsupported RoboMimic task suite '{task_name}'. Supported: {SUPPORTED_TASKS}")

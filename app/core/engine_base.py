"""Common interface both model engines (radicle, rootseg) implement, so the GUI/CLI/job runner
never need to know which model they're driving.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

# progress_cb(message, fraction_complete_or_None)
ProgressCB = Optional[Callable[[str, Optional[float]], None]]
# cancel_cb() -> True if the run should stop at the next safe point
CancelCB = Optional[Callable[[], bool]]


@dataclass
class RunResult:
    outputs: List[Path]
    device: str
    elapsed_s: float
    extra: Dict = field(default_factory=dict)


class Engine:
    key: str = ""
    display_name: str = ""

    def load(self, progress_cb: ProgressCB = None) -> None:
        """Load checkpoint(s) into memory. Safe to call once and reuse across many run() calls."""
        raise NotImplementedError

    def run(self, volume_path: Path, out_dir: Path, progress_cb: ProgressCB = None,
            cancel_cb: CancelCB = None) -> RunResult:
        raise NotImplementedError

"""Shared types for the model engine and the job runner / GUIs that drive it."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

# progress_cb(message, fraction_complete_or_None)
ProgressCB = Optional[Callable[[str, Optional[float]], None]]
# cancel_cb() -> True if the run should stop at the next safe point
CancelCB = Optional[Callable[[], bool]]


@dataclass
class LiveInfo:
    """Where a running volume's input and (partially written) prediction live, so a viewer can
    read any already-finished slice. Prediction slices < `done` are final."""
    input_path: Path
    pred_path: Path
    shape: tuple
    done: int  # highest slice index written so far


# slice_cb(live_info) -- called after every predicted slice
SliceCB = Optional[Callable[[LiveInfo], None]]


@dataclass
class RunResult:
    outputs: List[Path]
    device: str
    elapsed_s: float
    extra: Dict = field(default_factory=dict)

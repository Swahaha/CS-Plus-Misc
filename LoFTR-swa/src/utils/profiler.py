import torch
from pytorch_lightning.profilers import SimpleProfiler, PassThroughProfiler
from contextlib import contextmanager
from pytorch_lightning.utilities import rank_zero_only


class InferenceProfiler(SimpleProfiler):
    """
    This profiler records duration of actions with cuda.synchronize()
    Use this in test time. 
    """

    def __init__(self):
        super().__init__()
        self.start = rank_zero_only(self.start)
        self.stop = rank_zero_only(self.stop)
        self.summary = rank_zero_only(self.summary)

    @contextmanager
    def profile(self, action_name: str) -> None:
        try:
            torch.cuda.synchronize()
            self.start(action_name)
            yield action_name
        finally:
            torch.cuda.synchronize()
            self.stop(action_name)


def build_profiler(profiler_name):
    if profiler_name == 'simple':
        return SimpleProfiler()
    elif profiler_name == 'pass_through':
        return PassThroughProfiler()
    else:
        return None

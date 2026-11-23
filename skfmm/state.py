"""Resumable travel-time marching with fixed inputs."""

import operator

import numpy as np

from .cfmm import (cStateAdvance, cStateCheckpoint, cStateCreate, cStateFlags,
                   cStateRestore, cStateResult, cStateStats)
from .pfmm import post_process_result, pre_process_args


class TravelTimeState:
    """Pause and restore a complete-domain solve without changing its inputs.

    Inputs are copied. Checkpoints belong to one state and stay in memory.
    Early exit, narrow bands, periodic grids, and input updates are not
    supported by this API. Use one state per serial caller.
    """

    def __init__(self, phi, speed, dx=1.0, self_test=False, order=2):
        if np.ma.is_masked(speed):
            raise ValueError("put obstacle masks on phi, not speed")
        phi, dx, flags, _, _ = pre_process_args(phi, dx, 0.0, False)
        self._handle = cStateCreate(phi, speed, dx, flags, order, int(self_test))

    def advance(self, batches=None):
        """Accept up to batches equal-time groups; return True when complete.

        None finishes the solve. Zero leaves it unchanged. Each group includes
        all neighbor updates, so a checkpoint never splits a tied group.
        """
        if batches is None:
            limit = -1
        else:
            limit = operator.index(batches)
            if limit < 0:
                raise ValueError("batches must be nonnegative or None")
        return cStateAdvance(self._handle, limit)

    def result(self):
        """Return a copy, with every cell not yet accepted masked."""
        return post_process_result(cStateResult(self._handle))

    def flags(self):
        """Return a copy: 0=Far, 1=Narrow, 2=Frozen, 3=Mask."""
        return cStateFlags(self._handle)

    @property
    def stats(self):
        """Batch count, completion, and array bytes; excludes object overhead."""
        return cStateStats(self._handle)

    def checkpoint(self, max_bytes=None):
        """Copy the current march, including heap order and back pointers.

        max_bytes limits array storage for this checkpoint. It does not limit
        the working state or earlier checkpoints held by the caller.
        """
        if max_bytes is not None:
            limit = operator.index(max_bytes)
            if limit < 0:
                raise ValueError("max_bytes must be nonnegative")
            if self.stats["checkpoint_array_bytes"] > limit:
                raise MemoryError("checkpoint exceeds max_bytes")
        return cStateCheckpoint(self._handle)

    def restore(self, checkpoint):
        """Restore a checkpoint from this state, leaving the checkpoint intact."""
        cStateRestore(self._handle, checkpoint)

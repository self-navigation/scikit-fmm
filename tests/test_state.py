"""Paused and restored marches must match uninterrupted travel times."""

import gc

import numpy as np
import pytest
import skfmm


def assert_same(actual, expected):
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual), np.ma.getmaskarray(expected)
    )
    np.testing.assert_array_equal(np.ma.getdata(actual), np.ma.getdata(expected))


@pytest.mark.parametrize("shape", [(31,), (25, 31), (9, 11, 7)])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("random_speed", [False, True])
def test_multiple_checkpoints_match_full_solve(shape, order, random_speed):
    rng = np.random.default_rng(29)
    phi = np.ones(shape)
    center = tuple(n // 2 for n in shape)
    phi[center] = -1
    mask = rng.random(shape) < 0.08
    mask[tuple(slice(c - 1, c + 2) for c in center)] = False
    phi = np.ma.array(phi, mask=mask)
    speed = rng.uniform(0.2, 2, shape) if random_speed else np.ones(shape)
    dx = np.linspace(0.1, 0.3, len(shape))
    reference = skfmm.travel_time(phi, speed, dx=dx, order=order, self_test=True)
    state = skfmm.TravelTimeState(phi, speed, dx=dx, order=order, self_test=True)
    saved = []
    for batches in (0, 1, 7, 23):
        state.advance(batches)
        saved.append((state.checkpoint(), state.result(), state.flags(), state.stats))
    assert state.advance()
    assert_same(state.result(), reference)
    complete = state.checkpoint()
    for checkpoint, partial, flags, stats in reversed(saved):
        state.restore(checkpoint)
        assert_same(state.result(), partial)
        np.testing.assert_array_equal(state.flags(), flags)
        assert state.stats == stats
        while not state.advance(11):
            pass
        assert_same(state.result(), reference)
    state.restore(complete)
    assert state.advance(0) and state.advance(1)
    assert_same(state.result(), reference)


def test_pause_accepts_a_whole_tied_group():
    phi = np.ones((21, 21))
    phi[10, 10] = 0
    state = skfmm.TravelTimeState(phi, np.ones_like(phi), self_test=True)
    assert not state.advance(0)
    assert np.count_nonzero(state.flags() == 2) == 1
    assert not state.advance(1)
    assert state.stats["batches"] == 1
    flags = state.flags()
    assert np.count_nonzero(flags == 2) == 5
    assert all(flags[target] == 2 for target in [(10, 9), (10, 11), (9, 10), (11, 10)])
    assert np.count_nonzero(flags == 1) > 0


def test_result_and_inputs_do_not_share_working_storage():
    phi = np.ones((21, 25))
    phi[10, 12] = -1
    speed = np.ones_like(phi)
    mask = np.zeros_like(phi, dtype=bool)
    mask[0, :] = True
    phi = np.ma.array(phi, mask=mask)
    dx = np.array([0.1, 0.2])
    reference = skfmm.travel_time(phi, speed, dx=dx)
    state = skfmm.TravelTimeState(phi, speed, dx=dx)
    phi[:] = 100
    phi.mask[:] = True
    speed[:] = 0
    dx[:] = 10
    state.advance(3)
    state.result()[:] = -999
    state.flags()[:] = 3
    state.advance()
    assert_same(state.result(), reference)


def test_noncontiguous_inputs_are_copied_in_grid_order():
    phi = np.ones((23, 27))
    phi[10, 12] = -1
    speed = np.arange(phi.size, dtype=float).reshape(phi.shape) / phi.size + 0.5
    phi, speed = phi.T[::-1, :], speed.T[::-1, :]
    reference = skfmm.travel_time(phi.copy(), speed.copy())
    state = skfmm.TravelTimeState(phi, speed)
    state.advance()
    assert_same(state.result(), reference)


def test_disconnected_and_zero_speed_cells_keep_distinct_flags():
    phi = np.ones((21, 21))
    phi[10, 5] = -1
    mask = np.zeros_like(phi, dtype=bool)
    mask[:, 10] = True
    phi = np.ma.array(phi, mask=mask)
    speed = np.ones(phi.shape)
    speed[5, 5] = 0
    speed[6, 5] = -1
    speed[7, 5] = np.finfo(float).eps / 2
    state = skfmm.TravelTimeState(phi, speed)
    state.advance()
    assert_same(state.result(), skfmm.travel_time(phi, speed))
    flags = state.flags()
    assert np.all(flags[:, 10] == 3)
    assert np.all(flags[:, 11:] == 0)
    assert np.all(flags[5:8, 5] == 3)
    assert not np.any(flags == 1)


def test_all_cells_initialized_with_no_heap_entries():
    phi = np.zeros((3, 4))
    state = skfmm.TravelTimeState(phi, np.ones_like(phi))
    saved = state.checkpoint()
    assert state.stats["complete"] and state.stats["batches"] == 0
    assert state.advance()
    state.restore(saved)
    assert_same(state.result(), phi)


def test_checkpoint_ownership_and_array_budget():
    phi = np.ones((21, 21))
    phi[10, 10] = -1
    state = skfmm.TravelTimeState(phi, np.ones_like(phi))
    state.advance(4)
    stats = state.stats
    with pytest.raises(MemoryError, match="max_bytes"):
        state.checkpoint(max_bytes=stats["checkpoint_array_bytes"] - 1)
    saved = state.checkpoint(max_bytes=stats["checkpoint_array_bytes"])
    assert state.stats == stats
    assert stats["working_array_bytes"] > stats["checkpoint_array_bytes"] > phi.nbytes
    other = skfmm.TravelTimeState(phi, np.ones_like(phi))
    with pytest.raises(ValueError, match="another state"):
        other.restore(saved)
    del state
    gc.collect()
    with pytest.raises(ValueError, match="another state"):
        other.restore(saved)
    other.advance()
    assert_same(other.result(), skfmm.travel_time(phi, np.ones_like(phi)))


@pytest.mark.parametrize("bad_batches", [-1, -2, 1.5])
def test_bad_batch_limit_does_not_change_the_state(bad_batches):
    state = skfmm.TravelTimeState([-1, 1, 1, 1], [1, 1, 1, 1])
    before = state.stats
    with pytest.raises((ValueError, TypeError)):
        state.advance(bad_batches)
    assert state.stats == before
    assert state.advance()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phi": [1, 1, 1]},
        {"phi": [], "speed": []},
        {"phi": [np.nan, 1, -1]},
        {"speed": [1, np.inf, 1]},
        {"speed": [1, 1]},
        {"dx": 0},
        {"dx": np.nan},
        {"dx": [1, 1]},
        {"order": 0},
        {"order": 3},
        {"speed": np.ma.array([1, 1, 1], mask=[False, True, False])},
    ],
)
def test_invalid_inputs(kwargs):
    args = {"phi": [-1, 1, 1], "speed": [1, 1, 1], **kwargs}
    with pytest.raises(ValueError):
        skfmm.TravelTimeState(**args)


def test_exhausted_heap_ids_raise_after_pops():
    heap = skfmm.heap(2)
    heap.push(0, 0.0)
    heap.pop()
    heap.push(1, 1.0)
    heap.pop()
    with pytest.raises(RuntimeError, match="heap full"):
        heap.push(2, 2.0)


if __name__ == "__main__":
    pytest.main()

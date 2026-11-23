"""Changed-input replay must match a fresh full travel-time solve."""

import numpy as np
import pytest
import skfmm


def assert_same(actual, expected):
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual), np.ma.getmaskarray(expected)
    )
    np.testing.assert_array_equal(np.ma.getdata(actual), np.ma.getdata(expected))


def reference(phi, speed, mask, order=2, dx=0.1):
    return skfmm.travel_time(np.ma.array(phi, mask=mask), speed, order=order, dx=dx)


def tracked_state(phi, speed, mask, order=2, dx=0.1):
    return skfmm.TravelTimeState(
        np.ma.array(phi, mask=mask), speed, order=order, dx=dx,
        self_test=True, track_dependencies=True,
    )


def finish_with_checkpoints(state, stops=(7, 25, 50, 100)):
    saved = [state.checkpoint()]
    last = state.stats["batches"]
    for stop in stops:
        if stop <= last:
            continue
        state.advance(stop - last)
        saved.append(state.checkpoint())
        last = state.stats["batches"]
        if state.stats["complete"]:
            break
    state.advance()
    if state.stats["batches"] != last:
        saved.append(state.checkpoint())
    return saved


def line_inputs():
    phi = np.ones(161)
    phi[0] = -1
    return phi, np.ones(phi.shape), np.zeros(phi.shape, dtype=bool)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("change", ["faster", "slower", "add", "remove", "mixed"])
def test_line_edits_reuse_a_nonzero_prefix(order, change):
    phi, speed, mask = line_inputs()
    if change == "remove":
        mask[140] = True
    state = tracked_state(phi, speed, mask, order)
    saved = finish_with_checkpoints(state)
    new_speed, new_mask = speed.copy(), mask.copy()
    if change == "faster":
        new_speed[140] = 2.0
    elif change == "slower":
        new_speed[140] = 0.25
    elif change == "add":
        new_mask[140] = True
    elif change == "remove":
        new_mask[140] = False
    else:
        new_speed[125] = 2.0
        new_speed[130] = 0.3
        new_mask[145] = True
    result = state.replay(new_speed, new_mask, saved)
    assert result["status"] == "replay"
    assert result["reason"] == "checkpoint_replayed"
    assert result["changed_cells"] > 0
    assert 0 < result["reused_batches"] < result["first_affected_batch"]
    assert state.stats["batches"] == result["reused_batches"]
    for old in saved:
        with pytest.raises(ValueError):
            state.restore(old)
    state.advance()
    assert_same(state.result(), reference(phi, new_speed, new_mask, order))


@pytest.mark.parametrize("order", [1, 2])
def test_repeated_edits_and_earlier_edit_fallback(order):
    phi, speed, mask = line_inputs()
    state = tracked_state(phi, speed, mask, order)
    saved = finish_with_checkpoints(state)
    statuses = []
    changes = [(140, 2.0, False), (130, 1.0, True), (130, 0.4, False),
               (50, 0.3, False), (145, 1.0, True), (145, 2.0, False)]
    for cell, value, blocked in changes:
        speed[cell] = value
        mask[cell] = blocked
        result = state.replay(speed, mask, saved)
        statuses.append(result["status"])
        if result["status"] == "full_required":
            state = tracked_state(phi, speed, mask, order)
        saved = finish_with_checkpoints(state)
        assert_same(state.result(), reference(phi, speed, mask, order))
    assert statuses[0] == "replay"
    assert statuses[3] == "full_required"
    assert statuses.count("replay") >= 3


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("random_speed", [False, True])
def test_random_2d_mixed_updates(order, random_speed):
    rng = np.random.default_rng(71)
    phi = np.ones((21, 25))
    phi[10, 4] = -1
    mask = rng.random(phi.shape) < 0.06
    mask[8:13, 2:7] = False
    speed = rng.uniform(0.3, 1.7, phi.shape) if random_speed else np.ones(phi.shape)
    dx = (0.1, 0.15)
    state = tracked_state(phi, speed, mask, order, dx)
    saved = finish_with_checkpoints(state, (3, 10, 25, 50))
    replays = 0
    for step in range(10):
        rows = rng.integers(2, 19, 8)
        cols = rng.integers(12, 24, 8)
        speed[rows, cols] = rng.uniform(0.2, 2.0, len(rows))
        mask[rows[:3], cols[:3]] = ~mask[rows[:3], cols[:3]]
        result = state.replay(speed, mask, saved)
        if result["status"] == "full_required":
            state = tracked_state(phi, speed, mask, order, dx)
        elif result["status"] == "replay":
            replays += result["reused_batches"] > 0
        saved = finish_with_checkpoints(state, (3, 10, 25, 50))
        assert_same(state.result(), reference(phi, speed, mask, order, dx))
    assert replays > 0


@pytest.mark.parametrize("order", [1, 2])
def test_opening_a_masked_wall_connects_a_large_region(order):
    phi = np.ones((31, 61))
    phi[15, 5] = -1
    mask = np.zeros(phi.shape, dtype=bool)
    mask[:, 20:] = True
    speed = np.ones(phi.shape)
    state = tracked_state(phi, speed, mask, order)
    saved = finish_with_checkpoints(state, (1, 5, 15, 30))
    before_count = np.count_nonzero(~np.ma.getmaskarray(state.result()))
    new_mask = np.zeros(phi.shape, dtype=bool)
    new_mask[:, 20] = True
    new_mask[15, 20] = False
    result = state.replay(speed, new_mask, saved)
    assert result["status"] == "replay"
    assert result["reused_batches"] > 0
    state.advance()
    assert_same(state.result(), reference(phi, speed, new_mask, order))
    assert np.count_nonzero(~np.ma.getmaskarray(state.result())) > 2 * before_count


@pytest.mark.parametrize("order", [1, 2])
def test_forward_restore_keeps_reads_of_rejected_masked_neighbors(order):
    phi, speed, mask = line_inputs()
    mask[50] = True
    state = tracked_state(phi, speed, mask, order)
    saved = finish_with_checkpoints(state, (10, 30))
    completed_batches = state.stats["batches"]
    state.restore(saved[0])
    state.restore(saved[-1])
    assert state.stats["complete"]
    mask[50] = False
    before = state.result()
    rejected = state.replay(speed, mask, [saved[-1]])
    assert rejected["status"] == "full_required"
    assert rejected["reason"] == "no_safe_checkpoint"
    assert 0 < rejected["first_affected_batch"] <= completed_batches
    assert_same(state.result(), before)
    result = state.replay(speed, mask, saved)
    assert result["status"] == "replay"
    assert 0 < result["reused_batches"] < result["first_affected_batch"] <= completed_batches
    state.advance()
    assert_same(state.result(), reference(phi, speed, mask, order))


def test_unread_disconnected_edit_can_reuse_completed_checkpoint():
    phi, speed, mask = line_inputs()
    mask[[80, 120]] = True
    state = tracked_state(phi, speed, mask)
    saved = finish_with_checkpoints(state)
    completed_batches = state.stats["batches"]
    mask[120] = False
    result = state.replay(speed, mask, saved)
    assert result["status"] == "replay"
    assert result["first_affected_batch"] == -1
    assert result["reused_batches"] == completed_batches
    assert state.stats["complete"]
    assert state.flags()[120] == 0
    state.advance()
    assert_same(state.result(), reference(phi, speed, mask))


@pytest.mark.parametrize("change", ["source_speed", "initial_narrow_mask", "new_source"])
def test_initialization_changes_require_full_solve_and_preserve_old_state(change):
    phi, speed, mask = line_inputs()
    if change == "new_source":
        phi[120] = -1
        mask[119:122] = True
    state = tracked_state(phi, speed, mask)
    saved = finish_with_checkpoints(state)
    before, flags, stats = state.result(), state.flags(), state.stats
    if change == "source_speed":
        speed[0] = 2.0
    elif change == "initial_narrow_mask":
        mask[2] = True
    else:
        mask[119:121] = False
    result = state.replay(speed, mask, saved)
    assert result["status"] == "full_required"
    assert result["reason"] == "initialization_changed"
    assert_same(state.result(), before)
    np.testing.assert_array_equal(state.flags(), flags)
    assert state.stats == stats
    state.restore(saved[-1])
    assert_same(state.result(), before)


def test_missing_safe_checkpoint_leaves_old_state_unchanged():
    phi, speed, mask = line_inputs()
    state = tracked_state(phi, speed, mask)
    state.advance()
    saved = state.checkpoint()
    before, stats = state.result(), state.stats
    speed[140] = 2.0
    for candidates in ([], [saved]):
        result = state.replay(speed, mask, candidates)
        assert result["status"] == "full_required"
        assert result["reason"] == "no_safe_checkpoint"
        assert state.stats == stats
        assert_same(state.result(), before)
        state.restore(saved)


@pytest.mark.parametrize("blocked_by_speed", [False, True])
def test_unchanged_effective_mask_keeps_checkpoints(blocked_by_speed):
    phi, speed, mask = line_inputs()
    if blocked_by_speed:
        speed[140] = 0.0
    else:
        mask[140] = True
    state = tracked_state(phi, speed, mask)
    saved = finish_with_checkpoints(state)
    speed[140] = 3.0
    mask[140] = True
    info = state.replay(speed, mask, saved)
    assert info["status"] == "unchanged"
    assert info["changed_cells"] == 0
    state.restore(saved[-1])
    assert_same(state.result(), reference(phi, speed, mask))
    mask[140] = False
    info = state.replay(speed, mask, saved)
    assert info["status"] == "replay"
    state.advance()
    assert_same(state.result(), reference(phi, speed, mask))


@pytest.mark.parametrize("value", [0.0, np.finfo(float).eps / 2, np.finfo(float).eps])
def test_speed_threshold_changes_match_effective_masks(value):
    phi, speed, mask = line_inputs()
    state = tracked_state(phi, speed, mask)
    saved = finish_with_checkpoints(state)
    speed[140] = value
    info = state.replay(speed, mask, saved)
    assert info["status"] == "replay"
    state.advance()
    assert_same(state.result(), reference(phi, speed, mask))
    assert (state.flags()[140] == 3) == (value < np.finfo(float).eps)


def test_replay_copies_new_inputs_and_rejects_old_checkpoints():
    phi, speed, mask = line_inputs()
    state = tracked_state(phi, speed, mask)
    saved = finish_with_checkpoints(state)
    speed[140] = 2.0
    expected = reference(phi, speed, mask)
    assert state.replay(speed, mask, saved)["status"] == "replay"
    speed[:] = 0
    mask[:] = True
    state.advance()
    assert_same(state.result(), expected)
    with pytest.raises(ValueError):
        state.restore(saved[0])
    with pytest.raises(ValueError):
        state.replay(np.ones(phi.shape), np.zeros(phi.shape, dtype=bool), saved)


def test_replay_requires_a_complete_tracked_state():
    phi, speed, mask = line_inputs()
    plain = skfmm.TravelTimeState(phi, speed)
    plain.advance()
    with pytest.raises(ValueError):
        plain.replay(speed, mask, [plain.checkpoint()])
    state = tracked_state(phi, speed, mask)
    saved = state.checkpoint()
    with pytest.raises(ValueError):
        state.replay(speed, mask, [saved])
    state.advance()
    with pytest.raises(ValueError):
        state.replay(speed, mask, [plain.checkpoint()])


@pytest.mark.parametrize("invalid", ["speed_shape", "mask_shape", "nan", "masked_speed"])
def test_invalid_replay_inputs_do_not_damage_the_state(invalid):
    phi, speed, mask = line_inputs()
    state = tracked_state(phi, speed, mask)
    saved = finish_with_checkpoints(state)
    before, stats = state.result(), state.stats
    if invalid == "speed_shape":
        speed = speed[:-1]
    elif invalid == "mask_shape":
        mask = mask[:-1]
    elif invalid == "nan":
        speed[140] = np.nan
    else:
        speed = np.ma.array(speed, mask=np.arange(len(speed)) == 140)
    with pytest.raises(ValueError):
        state.replay(speed, mask, saved)
    assert state.stats == stats
    assert_same(state.result(), before)


def test_second_order_frozen_prefix_counterexamples():
    rng = np.random.default_rng(9)
    violations, reused = [], 0
    for case in range(250):
        mask = rng.random((31, 31)) < 0.07
        mask[15, 15] = False
        speed = 0.2 + 0.8 * rng.random(mask.shape)
        phi = np.ones(mask.shape)
        phi[15, 15] = -1
        cell = tuple(rng.integers(2, 29, 2))
        mask[cell] = False
        before = reference(phi, speed, mask, dx=0.05)
        changed = speed.copy()
        changed[cell] = max(0.1, speed[cell] * 0.2)
        expected = reference(phi, changed, mask, dx=0.05)
        t0, t1 = np.ma.filled(before, np.nan), np.ma.filled(expected, np.nan)
        old_prefix = (t0 < t0[cell] - 1e-10) & np.isfinite(t1)
        if not np.any(np.abs((t1 - t0)[old_prefix]) > 1e-8):
            continue
        violations.append(case)
        state = tracked_state(phi, speed, mask, dx=0.05)
        saved = finish_with_checkpoints(state, (1, 3, 7, 15, 31, 63, 127, 255))
        info = state.replay(changed, mask, saved)
        if info["status"] == "full_required":
            state = tracked_state(phi, changed, mask, dx=0.05)
        else:
            assert info["status"] == "replay"
            reused += info["reused_batches"] > 0
        state.advance()
        assert_same(state.result(), expected)
    assert 56 in violations
    assert reused > 0

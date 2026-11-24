"""Initialization checks must keep exact fields for supported and skipped edits."""

import numpy as np
import pytest
import skfmm


def tracked(phi, speed, mask, order=2, dx=0.1):
    return skfmm.TravelTimeState(
        np.ma.array(phi, mask=mask), speed, dx=dx, order=order,
        self_test=True, track_dependencies=True,
    )


def assert_same(actual, expected):
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))
    np.testing.assert_array_equal(np.ma.getdata(actual), np.ma.getdata(expected))


def assert_fresh(state, phi, speed, mask, order=2, dx=0.1):
    expected = skfmm.travel_time(
        np.ma.array(phi, mask=mask), speed, dx=dx, order=order, self_test=True,
    )
    assert_same(state.result(), expected)


@pytest.mark.parametrize("shape", [(41,), (13, 41), (5, 9, 41), (1,) * 9 + (41,)])
@pytest.mark.parametrize("order", [1, 2])
def test_distant_speed_edit_skips_initialization_and_matches_fresh_field(shape, order):
    phi = np.ones(shape)
    row = tuple(size // 2 for size in shape[:-1])
    phi[row + (3,)] = -1.0
    speed = np.ones(shape)
    mask = np.zeros(shape, dtype=bool)
    dx = np.linspace(0.1, 0.2, len(shape))
    state = tracked(phi, speed, mask, order, dx)
    initial = state.checkpoint()
    state.advance(5)
    middle = state.checkpoint()
    state.advance()

    speed[row + (35,)] = 0.6
    info = state.replay(speed, mask, [initial, middle])
    assert info["status"] == "replay"
    assert info["initialization_check"] == "unchanged_support"
    assert info["reason"] == "checkpoint_replayed"
    assert info["reused_batches"] > 0
    state.advance()
    assert_fresh(state, phi, speed, mask, order, dx)


@pytest.mark.parametrize("kind", ["zero", "sign_crossing"])
@pytest.mark.parametrize("order", [1, 2])
def test_unmasking_a_hidden_source_requires_full_initialization_comparison(kind, order):
    phi = np.ones(41)
    phi[0] = 0.0
    phi[27] = 0.0 if kind == "zero" else -1.0
    speed = np.ones(phi.shape)
    mask = np.zeros(phi.shape, dtype=bool)
    mask[25:30] = True
    state = tracked(phi, speed, mask, order)
    state.advance()
    before, flags, stats = state.result(), state.flags(), state.stats
    saved = state.checkpoint()

    mask[27] = False
    if kind == "sign_crossing":
        mask[28] = False
    info = state.replay(speed, mask, [])
    assert info["status"] == "full_required"
    assert info["reason"] == "initialization_changed"
    assert info["initialization_check"] == "full_comparison"
    assert info["first_affected_batch"] == -1
    assert state.stats == stats
    np.testing.assert_array_equal(state.flags(), flags)
    assert_same(state.result(), before)
    state.restore(saved)
    assert_same(state.result(), before)
    fresh = tracked(phi, speed, mask, order)
    fresh.advance()
    assert_fresh(fresh, phi, speed, mask, order)
    assert not np.array_equal(np.ma.getmaskarray(before), np.ma.getmaskarray(fresh.result()))


def test_conservative_support_hit_can_keep_the_complete_state():
    phi = np.ones(41)
    phi[0] = 0.0
    phi[27] = -1.0
    speed = np.ones(phi.shape)
    mask = np.zeros(phi.shape, dtype=bool)
    mask[24:31] = True
    state = tracked(phi, speed, mask)
    state.advance()
    batches = state.stats["batches"]

    # This cell neighbors a possible source, but that source stays masked.
    # The newly opened cell stays disconnected from the actual source.
    mask[25] = False
    info = state.replay(speed, mask, [])
    assert info["initialization_check"] == "full_comparison"
    assert info["reason"] == "complete_state_reused"
    assert state.stats["complete"] and state.stats["batches"] == batches
    assert state.flags()[25] == 0
    assert_fresh(state, phi, speed, mask)


@pytest.mark.parametrize("cell", [0, 2], ids=["frozen_source", "initial_narrow"])
@pytest.mark.parametrize("order", [1, 2])
def test_active_initial_speed_change_is_compared_and_rejected(cell, order):
    phi = np.ones(41)
    phi[0] = -1.0
    speed = np.ones(phi.shape)
    mask = np.zeros(phi.shape, dtype=bool)
    state = tracked(phi, speed, mask, order)
    saved = state.checkpoint()
    state.advance()
    before, flags, stats = state.result(), state.flags(), state.stats

    speed[cell] = 2.0
    info = state.replay(speed, mask, [saved])
    assert info["initialization_check"] == "full_comparison"
    assert info["reason"] == "initialization_changed"
    assert info["status"] == "full_required"
    assert state.stats == stats
    np.testing.assert_array_equal(state.flags(), flags)
    assert_same(state.result(), before)
    state.restore(saved)
    state.advance()
    assert_same(state.result(), before)


def test_repeated_unread_edits_keep_support_checks_and_restore_exact_results():
    phi = np.ones(41)
    phi[0] = -1.0
    speed = np.ones(phi.shape)
    mask = np.zeros(phi.shape, dtype=bool)
    mask[15] = True
    state = tracked(phi, speed, mask)
    state.advance()
    batches = state.stats["batches"]
    epsilon = np.finfo(float).eps

    for value in (1.4, 0.0, epsilon / 2, epsilon, 0.8):
        old = state.checkpoint()
        speed[30] = value
        info = state.replay(speed, mask, [])
        if value == epsilon / 2:
            assert info["status"] == "unchanged"
            assert info["initialization_check"] == "not_needed"
            state.restore(old)
        else:
            assert info["reason"] == "complete_state_reused"
            assert info["initialization_check"] == "unchanged_support"
            with pytest.raises(ValueError, match="input revision"):
                state.restore(old)
        assert state.stats["complete"] and state.stats["batches"] == batches
        assert (state.flags()[30] == 3) == (value < epsilon)
        assert_fresh(state, phi, speed, mask)
        saved, before = state.checkpoint(), state.result()
        state.restore(saved)
        assert_same(state.result(), before)


def test_ignored_masked_speed_and_missing_checkpoint_do_not_check_initialization():
    phi = np.ones(41)
    phi[0] = -1.0
    speed = np.ones(phi.shape)
    mask = np.zeros(phi.shape, dtype=bool)
    mask[30] = True
    state = tracked(phi, speed, mask)
    state.advance()
    before, stats = state.result(), state.stats
    saved = state.checkpoint()

    speed[30] = 2.0
    info = state.replay(speed, mask, [saved])
    assert info["status"] == "unchanged"
    assert info["initialization_check"] == "not_needed"
    state.restore(saved)
    assert_fresh(state, phi, speed, mask)

    speed[20] = 0.5
    info = state.replay(speed, mask, [])
    assert info["reason"] == "no_safe_checkpoint"
    assert info["initialization_check"] == "not_needed"
    assert state.stats == stats
    assert_same(state.result(), before)


def test_underflowed_phi_product_does_not_add_a_false_source_dependency():
    phi = np.ones(41)
    phi[0] = 0.0
    tiny = np.finfo(float).smallest_subnormal
    phi[29:32] = (tiny, -tiny, tiny)
    assert float(phi[29]) * float(phi[30]) == 0.0
    speed = np.ones(phi.shape)
    mask = np.zeros(phi.shape, dtype=bool)
    state = tracked(phi, speed, mask)
    saved = state.checkpoint()
    state.advance()

    speed[30] = 2.0
    info = state.replay(speed, mask, [saved])
    assert info["status"] == "replay"
    assert info["initialization_check"] == "unchanged_support"
    state.advance()
    assert_fresh(state, phi, speed, mask)


def test_dense_distant_edits_use_full_comparison_and_still_replay_exactly():
    phi = np.ones(101)
    phi[0] = -1.0
    speed = np.ones(phi.shape)
    mask = np.zeros(phi.shape, dtype=bool)
    state = tracked(phi, speed, mask)
    initial = state.checkpoint()
    state.advance()

    # Changing half the domain uses the bounded full comparison path, even
    # though every changed input is far from the initial source and band.
    speed[50:] = 1.5
    info = state.replay(speed, mask, [initial])
    assert info["initialization_check"] == "full_comparison"
    assert info["reason"] == "checkpoint_replayed"
    assert info["status"] == "replay"
    state.advance()
    assert_fresh(state, phi, speed, mask)

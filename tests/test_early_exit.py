"""Early exit must handle every accepted target, including initial seeds."""

import numpy as np
import pytest
import skfmm


def assert_matches_full(partial, full):
    reached = ~np.ma.getmaskarray(partial)
    np.testing.assert_array_equal(
        np.asarray(partial)[reached], np.asarray(full)[reached]
    )
    return reached


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("margin", [0, 3])
def test_targets_in_equal_time_batch(order, margin):
    phi = np.ones((41, 41))
    phi[20, 20] = -1
    speed = np.ones_like(phi)
    dx = 0.2
    full = skfmm.travel_time(phi, speed, dx=dx, order=order)
    counts = []
    for target in [(20, 30), (20, 10), (30, 20), (10, 20)]:
        partial = skfmm.travel_time(
            phi,
            speed,
            dx=dx,
            order=order,
            early_exit_target=target,
            early_exit_margin=margin,
        )
        reached = assert_matches_full(partial, full)
        assert reached[target]
        assert not reached.all()
        assert np.asarray(partial)[reached].max() <= full[target] + margin * dx
        counts.append(int(reached.sum()))
    assert len(set(counts)) == 1


@pytest.mark.parametrize("target", [(20, 20), (20, 21), (19, 20)])
@pytest.mark.parametrize("margin", [0, 2])
def test_target_already_frozen_at_initialization(target, margin):
    phi = np.ones((41, 41))
    phi[20, 20] = -1
    full = skfmm.travel_time(phi, np.ones_like(phi))
    partial = skfmm.travel_time(
        phi, np.ones_like(phi), early_exit_target=target, early_exit_margin=margin
    )
    reached = assert_matches_full(partial, full)
    assert reached[target]
    assert 5 <= reached.sum() < 100


def test_zero_contour_target_with_zero_margin():
    phi = np.ones((21, 21))
    phi[10, 10] = 0
    partial = skfmm.travel_time(phi, np.ones_like(phi), early_exit_target=(10, 10))
    assert np.count_nonzero(~np.ma.getmaskarray(partial)) == 1
    assert partial[10, 10] == 0


@pytest.mark.parametrize("target", [(10, 15), (10, 10)])
def test_unreachable_or_masked_target_finishes_reachable_component(target):
    phi = np.ones((21, 21))
    phi[10, 5] = -1
    mask = np.zeros_like(phi, dtype=bool)
    mask[:, 10] = True
    phi = np.ma.array(phi, mask=mask)
    full = skfmm.travel_time(phi, np.ones(phi.shape))
    partial = skfmm.travel_time(phi, np.ones(phi.shape), early_exit_target=target)
    np.testing.assert_array_equal(np.ma.getmaskarray(partial), np.ma.getmaskarray(full))
    assert_matches_full(partial, full)


def test_margin_is_travel_time_scaled_by_smallest_spacing():
    phi = np.ones((41, 41))
    phi[20, 20] = -1
    speed = np.full_like(phi, 0.4)
    spacing = (0.2, 0.3)
    target = (20, 30)
    margin = 3
    full = skfmm.travel_time(phi, speed, dx=spacing)
    partial = skfmm.travel_time(
        phi, speed, dx=spacing, early_exit_target=target, early_exit_margin=margin
    )
    reached = assert_matches_full(partial, full)
    assert reached[target] and not reached.all()
    assert np.asarray(partial)[reached].max() <= full[target] + margin * min(spacing)

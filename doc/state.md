# Resumable travel-time solver

`skfmm.TravelTimeState` uses the same first- or second-order updates as
`skfmm.travel_time`. It keeps the working grid and heap after each pause.
The initial implementation requires fixed inputs and a complete domain.

```python
state = skfmm.TravelTimeState(phi, speed, dx=0.05, order=2)
state.advance(1000)
saved = state.checkpoint(max_bytes=256 * 1024 * 1024)
partial = state.result()
state.advance()
full = state.result()

state.restore(saved)
state.advance()
replayed = state.result()
```

`full` and `replayed` must match exactly on the same build. They must also
match a fresh `travel_time` call with the same supported inputs.

## Operations

| Operation | Behavior |
| --- | --- |
| Constructor | Copy inputs and initialize source cells and the narrow band. |
| `advance(batches)` | Accept at most this many equal-time groups, including all neighbor updates. Return whether the domain is complete. |
| `advance()` | Finish the reachable domain. |
| `advance(0)` | Leave the state unchanged. |
| `result()` | Return a copy with cells not yet accepted masked. |
| `flags()` | Return a copy: Far=0, Narrow=1, Frozen=2, Mask=3. |
| `checkpoint(max_bytes=None)` | Copy the numerical state and heap without changing either. |
| `restore(saved)` | Restore a checkpoint from this instance. The checkpoint remains reusable. |
| `stats` | Return completed batch count, domain completion, and array storage sizes. |

Checkpoints preserve raw distances, flags, heap entries in their current
order, heap IDs, back pointers, batch count, and target-time state. They
do not rebuild or sort the heap. Input phi, speed, spacing, shape, and order
stay fixed in the owning state. The heap reserves room for the full grid.

Result export does not overwrite tentative distances. A Far cell after a
complete solve is disconnected; a Far cell during a paused solve may still
be reachable. Mask cells are kept separate from Far cells in `flags()`.

## Limits and ownership

- Supports finite inputs in 1 to 10 dimensions, positive spacing, and order
  1 or 2. As in `travel_time`, speeds below machine epsilon mark obstacles.
- Put obstacle masks on `phi`. Masked speed arrays are rejected.
- Early exit, narrow bands, periodic grids, and changed inputs are not
  supported. Their arguments are not part of this constructor.
- Inputs, returned arrays, and checkpoints own separate storage. A checkpoint
  can only be restored to the instance that created it.
- States and checkpoints are process-local and cannot be pickled. Deleting
  one releases its storage. A checkpoint does not keep the working state alive.
- Use one instance per serial caller. Computation runs in C++ and holds the
  Python GIL, as the existing stateless entry point does.
- A numerical exception marks the working state as failed. Restore a valid
  checkpoint or create a new state before using it again.

`working_array_bytes` includes the input copies, working arrays, and reserved
heap arrays. `checkpoint_array_bytes` is the array payload for a new checkpoint.
These counts exclude object headers and allocator overhead. `max_bytes`
limits the payload of one checkpoint; the caller must budget the working
state and all retained checkpoints separately. No automatic checkpoints are
kept by the solver.

This API does not yet implement incremental updates after an obstacle or
speed change. Such updates need dependency tracking and a checked restart
point before the first affected operation.

## Tests

Build and install this checkout before running the tests from outside the
source directory:

```bash
python -m pytest /path/to/scikit-fmm/tests
python -c 'import skfmm; raise SystemExit(skfmm.test())'
```

The state tests compare reached masks and values exactly, with first- and
second-order updates, uniform and random speeds, disconnected maps, ties,
noncontiguous inputs, repeated restores, and heap self-checks.

The native test also supports memory checks. Run from the checkout root:

```bash
c++ -std=c++11 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -I skfmm tests/test_state_native.cpp \
  skfmm/heap.cpp skfmm/base_marcher.cpp skfmm/distance_marcher.cpp \
  skfmm/travel_time_marcher.cpp -o /tmp/test-fmm-state
/tmp/test-fmm-state
```

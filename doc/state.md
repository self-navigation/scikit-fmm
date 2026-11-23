# Resumable travel-time solver

`skfmm.TravelTimeState` uses the same first- or second-order updates as
`skfmm.travel_time`. It keeps the working grid and heap after each pause.
It supports fixed-input checkpoints and optional checked replay after speed
and obstacle changes. Both modes solve the complete reachable domain.

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
| `replay(speed, mask, checkpoints)` | Check new inputs and restore a safe checkpoint, or report that a full solve is required. Requires dependency tracking and a complete state. |
| `stats` | Return completed batch count, domain completion, and array storage sizes. |

Checkpoints preserve raw distances, flags, heap entries in their current
order, heap IDs, back pointers, batch count, and target-time state. They
do not rebuild or sort the heap. Input phi values, spacing, shape, and order
stay fixed in the owning state. The heap reserves room for the full grid.

Result export does not overwrite tentative distances. A Far cell after a
complete solve is disconnected; a Far cell during a paused solve may still
be reachable. Mask cells are kept separate from Far cells in `flags()`.

## Speed and obstacle updates

Enable tracking when creating the state. Save checkpoints during the solve;
checkpoints taken earlier allow more changes to reuse work.

```python
state = skfmm.TravelTimeState(phi, speed, dx=0.05, order=2,
                            track_dependencies=True)
checkpoints = [state.checkpoint()]
while not state.advance(1000):
    checkpoints.append(state.checkpoint())
checkpoints.append(state.checkpoint())

info = state.replay(new_speed, new_mask, checkpoints)
if info["status"] == "full_required":
    new_phi = np.ma.array(np.ma.getdata(phi), mask=new_mask)
    state = skfmm.TravelTimeState(new_phi, new_speed,
                                dx=0.05, order=2, track_dependencies=True)
if info["status"] != "unchanged":
    checkpoints = [state.checkpoint()]
    while not state.advance(1000):
        checkpoints.append(state.checkpoint())
    checkpoints.append(state.checkpoint())
updated = state.result()
```

`new_mask` is a full array with the same shape as phi; true means blocked.
It replaces the explicit obstacle mask. Speed values below machine epsilon
also block cells. Changes to speed in cells blocked in both inputs do not
count as effective changes. All inputs are copied. Masked speed arrays are
rejected; supply the obstacle mask separately.

The result is a dictionary with these fields:

| Field | Meaning |
| --- | --- |
| `status` | `replay`, `unchanged`, or `full_required`. |
| `reason` | `checkpoint_replayed`, `inputs_unchanged`, `initialization_changed`, or `no_safe_checkpoint`. |
| `changed_cells` | Number of cells with different effective inputs. |
| `reused_batches` | Number of accepted batches in the selected checkpoint; for unchanged inputs, the current batch count. |
| `first_affected_batch` | First runtime batch that read a changed cell, or -1 if none did. |

`replay` installs the new speed and mask at the selected checkpoint. It does
not march further; call `advance()` or collect new checkpoints while advancing.
Every checkpoint from before a successful replay becomes invalid, including
the selected checkpoint. `unchanged` keeps the state and all checkpoints.
`full_required` also keeps the old state and checkpoints; create a new state
to solve the changed inputs. Invalid inputs, stale or foreign checkpoints,
an incomplete state, and disabled tracking raise `ValueError`.

Tracking records the first runtime read of each cell, including reads that
reject a masked neighbor. A safe checkpoint has fewer completed batches than
the first affected read. The solver checks that fresh initialization with the
new inputs has the same source and initial narrow-band state. It also checks
that changed cells in the checkpoint are Far or Mask and have no heap entry.
These checks apply to speed increases, decreases, and obstacle additions or
removals. They do not depend on an assumed order of arrival-time changes.
An edit in an unread disconnected region may reuse a completed checkpoint.

Ordinary `restore()` keeps accumulated read history for the current inputs,
even when restoring backward and then forward. Successful replay starts a new
checkpoint generation and keeps only the dependency history of the reused
prefix. On the same build, finishing a replay must match a fresh `travel_time`
solve exactly in reached masks and values.

## Limits and ownership

- Supports finite inputs in 1 to 10 dimensions, positive spacing, and order
  1 or 2. As in `travel_time`, speeds below machine epsilon mark obstacles.
- Put obstacle masks on `phi`. Masked speed arrays are rejected.
- Early exit, narrow bands, and periodic grids are not supported. Their
  arguments are not part of this constructor.
- Tracking is off by default. Fixed-input use does not allocate dependency
  history; enable `track_dependencies=True` before the first solve to replay.
- Inputs, returned arrays, and checkpoints own separate storage. A checkpoint
  can only be restored to the instance that created it.
- States and checkpoints are process-local and cannot be pickled. Deleting
  one releases its storage. A checkpoint does not keep the working state alive.
- Use one instance per serial caller. Computation runs in C++ and holds the
  Python GIL, as the existing stateless entry point does.
- A numerical exception marks the working state as failed. Restore a valid
  checkpoint or create a new state before using it again.

`working_array_bytes` includes the input copies, working arrays, and reserved
heap arrays. With tracking enabled, it also includes dependency history and
the saved initialization signature. `checkpoint_array_bytes` is the array
payload for a new checkpoint.
These counts exclude object headers and allocator overhead. `max_bytes`
limits the payload of one checkpoint; the caller must budget the working
state and all retained checkpoints separately. No automatic checkpoints are
kept by the solver.

## Tests

Build and install this checkout before running the tests from outside the
source directory:

```bash
python -m pytest /path/to/scikit-fmm/tests
python -c 'import skfmm; raise SystemExit(skfmm.test())'
```

The state tests compare reached masks and values exactly, with first- and
second-order updates, uniform and random speeds, disconnected maps, ties,
noncontiguous inputs, repeated restores, and heap self-checks. Replay tests
cover speed and obstacle changes, repeated edits, source initialization,
checkpoint invalidation, unread regions, and second-order cases where
arrival times alone cannot identify a safe prefix.

The native test also supports memory checks. Run from the checkout root:

```bash
c++ -std=c++11 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -I skfmm tests/test_state_native.cpp \
  skfmm/heap.cpp skfmm/base_marcher.cpp skfmm/distance_marcher.cpp \
  skfmm/travel_time_marcher.cpp -o /tmp/test-fmm-state
/tmp/test-fmm-state
```

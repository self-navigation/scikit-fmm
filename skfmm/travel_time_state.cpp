#define PY_ARRAY_UNIQUE_SYMBOL SKFMM_ARRAY_API
#define NO_IMPORT_ARRAY
#include "Python.h"
#include "numpy/ndarrayobject.h"
#include "travel_time_state.h"
#include "travel_time_marcher.h"

#include <climits>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
const char *state_name = "skfmm.TravelTimeState";
const char *checkpoint_name = "skfmm.TravelTimeCheckpoint";

struct State {
  std::vector<double> phi, speed, dx, distance;
  std::vector<long long> flags;
  std::vector<int> shape;
  std::vector<long long> first_read;
  baseMarcher::Initialization initial;
  int order = 2;
  bool self_test = false;
  bool track_dependencies = false;
  std::shared_ptr<int> owner = std::make_shared<int>(0);
  bool failed = false;
  std::unique_ptr<travelTimeMarcher> marcher;
};

struct Checkpoint {
  std::shared_ptr<int> owner;
  baseMarcher::Snapshot data;
};

struct ArrayDeleter {
  void operator()(PyArrayObject *array) const { Py_XDECREF(array); }
};
using Array = std::unique_ptr<PyArrayObject, ArrayDeleter>;

struct ObjectDeleter {
  void operator()(PyObject *object) const { Py_XDECREF(object); }
};
using Object = std::unique_ptr<PyObject, ObjectDeleter>;

Array as_array(PyObject *object, int dtype, int min_dim, int max_dim) {
  return Array(reinterpret_cast<PyArrayObject *>(
      PyArray_FromAny(object, PyArray_DescrFromType(dtype), min_dim, max_dim,
                      NPY_ARRAY_CARRAY_RO, NULL)));
}

void delete_state(PyObject *capsule) {
  delete static_cast<State *>(PyCapsule_GetPointer(capsule, state_name));
}

void delete_checkpoint(PyObject *capsule) {
  delete static_cast<Checkpoint *>(PyCapsule_GetPointer(capsule, checkpoint_name));
}

State *get_state(PyObject *capsule, bool allow_failed = false) {
  State *state = static_cast<State *>(PyCapsule_GetPointer(capsule, state_name));
  if (state && state->failed && !allow_failed) {
    PyErr_SetString(PyExc_RuntimeError, "failed state: restore a checkpoint or create a new state");
    return NULL;
  }
  return state;
}

PyObject *exception_to_python() {
  try {
    throw;
  } catch (const std::bad_alloc &) {
    return PyErr_NoMemory();
  } catch (const std::invalid_argument &error) {
    PyErr_SetString(PyExc_ValueError, error.what());
  } catch (const std::exception &error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "unknown error in travel-time state");
  }
  return NULL;
}

PyArrayObject *output_array(const State &state, int dtype) {
  npy_intp shape[MaximumDimension];
  for (std::size_t i = 0; i < state.shape.size(); i++)
    shape[i] = state.shape[i];
  return reinterpret_cast<PyArrayObject *>(
      PyArray_SimpleNew(state.shape.size(), shape, dtype));
}

bool same_shape(PyArrayObject *array, const State &state) {
  if (PyArray_NDIM(array) != static_cast<int>(state.shape.size()))
    return false;
  for (std::size_t i = 0; i < state.shape.size(); i++)
    if (PyArray_DIM(array, i) != state.shape[i])
      return false;
  return true;
}

bool same_initialization(State &state, double *speed, const std::vector<char> &mask) {
  std::vector<long long> flags(mask.size());
  std::vector<double> distance(mask.size(), 0.0);
  for (std::size_t i = 0; i < mask.size(); i++)
    flags[i] = mask[i] ? Mask : Far;
  travelTimeMarcher initial(state.phi.data(), state.dx.data(), flags.data(),
                            distance.data(), state.shape.size(), state.shape.data(),
                            state.self_test, state.order, speed, 0.0, 0);
  initial.initialize(true);
  return initial.getError() == 0 && initial.initialization() == state.initial;
}

PyObject *replay_report(const char *status, const char *reason, std::size_t changed,
                        std::size_t reused, long long first_affected,
                        const char *initialization_check = "not_needed") {
  return Py_BuildValue("{s:s,s:s,s:K,s:K,s:L,s:s}",
                       "status", status, "reason", reason,
                       "changed_cells", static_cast<unsigned long long>(changed),
                       "reused_batches", static_cast<unsigned long long>(reused),
                       "first_affected_batch", first_affected,
                       "initialization_check", initialization_check);
}
} // namespace

PyObject *state_create(PyObject *, PyObject *args) {
  PyObject *pphi, *pspeed, *pdx, *pflags;
  int order, self_test, track_dependencies = 0;
  if (!PyArg_ParseTuple(args, "OOOOii|p", &pphi, &pspeed, &pdx, &pflags,
                        &order, &self_test, &track_dependencies))
    return NULL;
  Array phi = as_array(pphi, NPY_DOUBLE, 1, 10);
  if (!phi)
    return NULL;
  Array speed = as_array(pspeed, NPY_DOUBLE, 1, 10);
  if (!speed)
    return NULL;
  Array dx = as_array(pdx, NPY_DOUBLE, 1, 1);
  if (!dx)
    return NULL;
  Array flags = as_array(pflags, NPY_LONGLONG, 1, 10);
  if (!flags)
    return NULL;
  try {
    if (order != 1 && order != 2)
      throw std::invalid_argument("order must be 1 or 2");
    if (self_test != 0 && self_test != 1)
      throw std::invalid_argument("self_test must be 0 or 1");
    if (!PyArray_SAMESHAPE(phi.get(), speed.get()) || !PyArray_SAMESHAPE(phi.get(), flags.get()))
      throw std::invalid_argument("phi, speed and flags must have the same shape");
    int ndim = PyArray_NDIM(phi.get());
    npy_intp size = PyArray_SIZE(phi.get());
    if (size == 0 || size > INT_MAX)
      throw std::invalid_argument("grid size must be between 1 and INT_MAX");
    if (PyArray_SIZE(dx.get()) != ndim)
      throw std::invalid_argument("dx must have one value per dimension");

    std::unique_ptr<State> state(new State);
    state->order = order;
    state->self_test = self_test;
    state->track_dependencies = track_dependencies;
    const double *phi_data = static_cast<double *>(PyArray_DATA(phi.get()));
    const double *speed_data = static_cast<double *>(PyArray_DATA(speed.get()));
    const double *dx_data = static_cast<double *>(PyArray_DATA(dx.get()));
    const long long *flag_data = static_cast<long long *>(PyArray_DATA(flags.get()));
    state->phi.assign(phi_data, phi_data + size);
    state->speed.assign(speed_data, speed_data + size);
    state->dx.assign(dx_data, dx_data + ndim);
    state->flags.assign(flag_data, flag_data + size);
    state->distance.resize(size, 0.0);
    for (int i = 0; i < ndim; i++) {
      state->shape.push_back(PyArray_DIM(phi.get(), i));
      if (!std::isfinite(dx_data[i]) || dx_data[i] <= 0)
        throw std::invalid_argument("dx must be finite and positive");
    }
    for (npy_intp i = 0; i < size; i++) {
      if (!std::isfinite(phi_data[i]) || !std::isfinite(speed_data[i]))
        throw std::invalid_argument("phi and speed must be finite");
      if (flag_data[i] != Far && flag_data[i] != Mask)
        throw std::invalid_argument("input flags must be Far or Mask");
    }
    state->marcher.reset(new travelTimeMarcher(
        state->phi.data(), state->dx.data(), state->flags.data(), state->distance.data(),
        ndim, state->shape.data(), self_test, order, state->speed.data(), 0.0, 0));
    state->marcher->initialize(true);
    if (state->marcher->getError() != 0)
      throw std::invalid_argument("phi contains no zero contour");
    if (state->track_dependencies) {
      state->initial = state->marcher->initialization();
      state->first_read.assign(size, -1);
      state->marcher->setDependencies(state->first_read.data());
    }
    PyObject *capsule = PyCapsule_New(state.get(), state_name, delete_state);
    if (capsule)
      state.release();
    return capsule;
  } catch (...) {
    return exception_to_python();
  }
}

PyObject *state_advance(PyObject *, PyObject *args) {
  PyObject *capsule;
  int batches;
  if (!PyArg_ParseTuple(args, "Oi", &capsule, &batches))
    return NULL;
  State *state = get_state(capsule);
  if (!state)
    return NULL;
  if (batches < -1) {
    PyErr_SetString(PyExc_ValueError, "batches must be nonnegative or -1");
    return NULL;
  }
  try {
    return PyBool_FromLong(state->marcher->advance(batches));
  } catch (...) {
    state->failed = true;
    return exception_to_python();
  }
}

PyObject *state_result(PyObject *, PyObject *capsule) {
  State *state = get_state(capsule);
  if (!state)
    return NULL;
  PyArrayObject *result = output_array(*state, NPY_DOUBLE);
  if (!result)
    return NULL;
  double *data = static_cast<double *>(PyArray_DATA(result));
  for (std::size_t i = 0; i < state->distance.size(); i++)
    data[i] = state->flags[i] == Frozen ? state->distance[i] : maxDouble;
  return reinterpret_cast<PyObject *>(result);
}

PyObject *state_flags(PyObject *, PyObject *capsule) {
  State *state = get_state(capsule);
  if (!state)
    return NULL;
  PyArrayObject *result = output_array(*state, NPY_LONGLONG);
  if (!result)
    return NULL;
  std::copy(state->flags.begin(), state->flags.end(),
            static_cast<long long *>(PyArray_DATA(result)));
  return reinterpret_cast<PyObject *>(result);
}

PyObject *state_checkpoint(PyObject *, PyObject *capsule) {
  State *state = get_state(capsule);
  if (!state)
    return NULL;
  try {
    std::unique_ptr<Checkpoint> saved(new Checkpoint);
    saved->owner = state->owner;
    saved->data = state->marcher->checkpoint();
    PyObject *result = PyCapsule_New(saved.get(), checkpoint_name, delete_checkpoint);
    if (result)
      saved.release();
    return result;
  } catch (...) {
    return exception_to_python();
  }
}

PyObject *state_restore(PyObject *, PyObject *args) {
  PyObject *capsule, *saved_capsule;
  if (!PyArg_ParseTuple(args, "OO", &capsule, &saved_capsule))
    return NULL;
  State *state = get_state(capsule, true);
  if (!state)
    return NULL;
  Checkpoint *saved = static_cast<Checkpoint *>(PyCapsule_GetPointer(saved_capsule, checkpoint_name));
  if (!saved)
    return NULL;
  if (saved->owner != state->owner) {
    PyErr_SetString(PyExc_ValueError, "checkpoint belongs to another state or input revision");
    return NULL;
  }
  try {
    state->marcher->restore(saved->data);
    state->failed = false;
    Py_RETURN_NONE;
  } catch (...) {
    state->failed = true;
    return exception_to_python();
  }
}

PyObject *state_stats(PyObject *, PyObject *capsule) {
  State *state = get_state(capsule);
  if (!state)
    return NULL;
  std::size_t working = state->marcher->arrayBytes()
      + (state->phi.size() + state->speed.size() + state->distance.size() + state->dx.size()) * sizeof(double)
      + state->flags.size() * sizeof(long long) + state->shape.size() * sizeof(int)
      + state->first_read.size() * sizeof(long long) + state->initial.arrayBytes();
  return Py_BuildValue("{s:K,s:O,s:K,s:K,s:O}",
                       "batches", static_cast<unsigned long long>(state->marcher->batchCount()),
                       "complete", state->marcher->complete() ? Py_True : Py_False,
                       "working_array_bytes", static_cast<unsigned long long>(working),
                       "checkpoint_array_bytes", static_cast<unsigned long long>(state->marcher->checkpointBytes()),
                       "track_dependencies", state->track_dependencies ? Py_True : Py_False);
}

PyObject *state_replay(PyObject *, PyObject *args) {
  PyObject *capsule, *pspeed, *pmask, *pcheckpoints;
  if (!PyArg_ParseTuple(args, "OOOO", &capsule, &pspeed, &pmask, &pcheckpoints))
    return NULL;
  State *state = get_state(capsule);
  if (!state)
    return NULL;
  if (!state->track_dependencies || !state->marcher->complete()) {
    PyErr_SetString(PyExc_ValueError, "replay requires a complete state with dependency tracking");
    return NULL;
  }
  Array speed = as_array(pspeed, NPY_DOUBLE, 1, 10);
  if (!speed)
    return NULL;
  Array mask = as_array(pmask, NPY_BOOL, 1, 10);
  if (!mask)
    return NULL;
  Object checkpoints(PySequence_Fast(pcheckpoints, "checkpoints must be a sequence"));
  if (!checkpoints)
    return NULL;

  bool mutating = false;
  try {
    if (!same_shape(speed.get(), *state) || !same_shape(mask.get(), *state))
      throw std::invalid_argument("speed and mask must match the state shape");
    std::vector<Checkpoint *> candidates;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(checkpoints.get()); i++) {
      PyObject *item = PySequence_Fast_GET_ITEM(checkpoints.get(), i);
      Checkpoint *saved = static_cast<Checkpoint *>(PyCapsule_GetPointer(item, checkpoint_name));
      if (!saved)
        return NULL;
      if (saved->owner != state->owner)
        throw std::invalid_argument("checkpoint belongs to another state or input revision");
      candidates.push_back(saved);
    }

    double *next_speed = static_cast<double *>(PyArray_DATA(speed.get()));
    const npy_bool *mask_data = static_cast<npy_bool *>(PyArray_DATA(mask.get()));
    std::vector<char> next_mask(state->flags.size());
    std::vector<int> changed;
    long long first_affected = -1;
    for (std::size_t i = 0; i < next_mask.size(); i++) {
      if (!std::isfinite(next_speed[i]))
        throw std::invalid_argument("speed must be finite");
      next_mask[i] = mask_data[i] || next_speed[i] < doubleEpsilon;
      bool old_mask = state->flags[i] == Mask;
      if (old_mask != static_cast<bool>(next_mask[i]) ||
          (!next_mask[i] && state->speed[i] != next_speed[i])) {
        changed.push_back(i);
        long long read = state->first_read[i];
        if (read >= 0 && (first_affected < 0 || read < first_affected))
          first_affected = read;
      }
    }
    if (changed.empty())
      return replay_report("unchanged", "inputs_unchanged", 0,
                            state->marcher->batchCount(), -1);
    bool complete_reuse = first_affected < 0;
    if (complete_reuse)
      for (int i : changed)
        if (!state->marcher->isInert(i)) {
          complete_reuse = false;
          break;
        }
    if (candidates.empty() && !complete_reuse)
      return replay_report("full_required", "no_safe_checkpoint", changed.size(), 0, first_affected);
    // A local support query visits at most 1 + 2*ndim possible source cells.
    // For dense edits, use the full comparison without repeated local scans.
    bool compare_initialization = changed.size() > state->phi.size() / (1 + 2 * state->shape.size());
    if (!compare_initialization)
      for (int i : changed)
        if (state->marcher->initializationMayRead(i)) {
          compare_initialization = true;
          break;
        }
    const char *initialization_check = compare_initialization ? "full_comparison" : "unchanged_support";
    if (compare_initialization && !same_initialization(*state, next_speed, next_mask))
      return replay_report("full_required", "initialization_changed", changed.size(), 0,
                            first_affected, initialization_check);

    Checkpoint *selected = NULL;
    if (!complete_reuse) {
      for (Checkpoint *candidate : candidates) {
        const auto &saved = candidate->data;
        if (saved.batches > state->marcher->batchCount() ||
            (first_affected >= 0 && saved.batches >= static_cast<std::size_t>(first_affected)))
          continue;
        bool inert = true;
        for (int i : changed)
          if ((saved.flags[i] != Far && saved.flags[i] != Mask) || saved.heapptr[i] != -1) {
            inert = false;
            break;
          }
        if (inert && (!selected || saved.batches > selected->data.batches))
          selected = candidate;
      }
      if (!selected)
        return replay_report("full_required", "no_safe_checkpoint", changed.size(), 0,
                              first_affected, initialization_check);
    }

    std::shared_ptr<int> next_owner = std::make_shared<int>(0);
    std::size_t reused = complete_reuse ? state->marcher->batchCount() : selected->data.batches;
    const char *reason = complete_reuse ? "complete_state_reused" : "checkpoint_replayed";
    Object result(replay_report("replay", reason, changed.size(), reused, first_affected,
                                initialization_check));
    if (!result)
      return NULL;
    mutating = true;
    if (!complete_reuse)
      state->marcher->restore(selected->data);
    std::copy(next_speed, next_speed + state->speed.size(), state->speed.begin());
    for (int i : changed) {
      state->flags[i] = next_mask[i] ? Mask : Far;
      state->distance[i] = 0.0;
    }
    if (!complete_reuse)
      for (long long &read : state->first_read)
        if (read > static_cast<long long>(reused))
          read = -1;
    state->owner.swap(next_owner);
    return result.release();
  } catch (...) {
    if (mutating)
      state->failed = true;
    return exception_to_python();
  }
}

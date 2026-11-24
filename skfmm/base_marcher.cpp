// fast_marching.cpp

// this file is a C++ implementation of the fast marching method. The
// code in this file is independent of Python.

#include "base_marcher.h"
#include "heap.h"
#include "math.h"
#include <vector>
#include <algorithm>
#include <stdexcept>

extern "C" {

baseMarcher::baseMarcher(double *phi, double *dx, long long *flag,
                         double *distance, int ndim, int *shape, bool self_test,
                         int order, double narrow, int periodic) {
  narrow_ = narrow;
  order_ = order;
  error_ = 1;
  phi_ = phi;
  dx_ = dx;
  flag_ = flag;
  distance_ = distance;
  dim_ = ndim;
  size_ = 1;
  self_test_ = self_test;
  heapptr_ = 0;
  heap_ = 0;
  periodic_ = periodic;
  early_exit_target_ = -1;
  early_exit_margin_cells_ = 0;
  early_exit_margin_T_ = 0.0;
  target_T_ = maxDouble;
  batches_ = 0;
  complete_ = false;
  first_read_ = 0;

  for (int i = 0; i < dim_; i++) {
    shape_[i] = shape[i];
    size_ *= shape[i];
    idx2_[i] = 1 / dx_[i] / dx_[i];
  }
  for (int i = 0; i < dim_; i++) {
    int prod = 1;
    for (int j = i + 1; j < dim_; j++)
      prod *= shape_[j];
    shift_[i] = prod;
  }
}

void baseMarcher::march() {
  initialize();
  if (error_ != 0)
    return;
  advance();
  for (int i = 0; i < size_; i++)
    if (flag_[i] != Frozen)
      distance_[i] = maxDouble;
  cleanUp();
}

void baseMarcher::initialize(bool reserve_grid) {
  if (heap_)
    throw std::logic_error("marcher is already initialized");
  initalizeFrozen();
  if (early_exit_target_ >= 0 && flag_[early_exit_target_] == Frozen)
    target_T_ = fabs(distance_[early_exit_target_]);

  int maxHeap = 0;
  int frozenCount = 0;
  for (int i = 0; i < size_; i++) {
    if (flag_[i] == Far)
      maxHeap++;
    if (flag_[i] == Frozen)
      frozenCount++;
  }
  if (!frozenCount) {
    error_ = 2;
    return;
  }
  heap_ = new heap(reserve_grid ? size_ : maxHeap, self_test_);
  heapptr_ = new int[size_];
  std::fill(heapptr_, heapptr_ + size_, -1);
  initalizeNarrow();
  complete_ = heap_->empty();
  error_ = 0;
}

baseMarcher::~baseMarcher() {
  delete heap_;
  delete[] heapptr_;
}

void baseMarcher::setEarlyExit(int target_addr, int margin_cells) {
  early_exit_target_ = target_addr;
  early_exit_margin_cells_ = margin_cells;

  // This is a travel-time margin, not a physical radius.
  double min_dx = dx_[0];
  for (int d = 1; d < dim_; d++)
    if (dx_[d] < min_dx)
      min_dx = dx_[d];
  early_exit_margin_T_ = margin_cells * min_dx;
}

void baseMarcher::initalizeNarrow() {
  // for each point in the far field check if neighbor is frozen
  // if so calculate distance and insert into heap
  for (int i = 0; i < size_; i++) {
    if (flag_[i] == Far) {
      for (int dim = 0; dim < dim_; dim++) {
        for (int j = -1; j < 2; j += 2) // each direction
        {
          int naddr = _getN(i, dim, j, Mask);
          if (naddr != -1 && flag_[naddr] == Frozen)
            if (flag_[i] == Far) {
              flag_[i] = Narrow;
              double d;
              if (order_ == 2)
                d = updatePointOrderTwo(i);
              else
                d = updatePointOrderOne(i);

              distance_[i] = d;
              heapptr_[i] = heap_->push(i, fabs(d));
            }
        } // for each direction
      } // for each dimension
    } // each far field point
  }
}

// now we apply the fast marching algorithm main loop
// (1) take the smallest narrow band element and
//     freeze it.
// (2) for each neighbor of the frozen point calculate distance based
// on frozen elements
//     - mark each neighbor as narrow band and stick it
//       into the heap
//     - if the neighbor is already in the heap update the
//       distance value.
bool baseMarcher::advance(int batch_limit) {
  if (!heap_ || error_ != 0)
    throw std::logic_error("marcher is not initialized");
  if (batch_limit < -1)
    throw std::invalid_argument("batch limit must be nonnegative or -1");
  if (complete_)
    return true;
  int i = 0;
  std::vector<int> toFreeze;

  while (!heap_->empty()) {
    if (batch_limit >= 0 && i == batch_limit)
      return false;
    toFreeze.clear();
    i++;
    double value = heap_->peek();
    int addr = 0;

    if ((narrow_ != 0) && (fabs(value) > narrow_)) {
      complete_ = true;
      break;
    }

    if (target_T_ != maxDouble &&
        fabs(value) > target_T_ + early_exit_margin_T_) {
      complete_ = true;
      break;
    }

    heap_->pop(&addr, &value);
    flag_[addr] = Frozen;
    finalizePoint(addr, value);
    toFreeze.push_back(addr);

    bool done = false;
    while (!done) {
      if (!heap_->empty() && value == heap_->peek()) {
        double l_value = 0;
        int l_addr = 0;
        heap_->pop(&l_addr, &l_value);
        flag_[l_addr] = Frozen;
        finalizePoint(l_addr, l_value);
        toFreeze.push_back(l_addr);
      } else
        done = true;
    }

    for (unsigned int k = 0; k < toFreeze.size(); k++) {
      int addr = toFreeze[k];
      if (addr == early_exit_target_)
        target_T_ = fabs(distance_[addr]);

      for (int dim = 0; dim < dim_; dim++) {
        for (int j = -1; j < 2; j += 2) // each direction
        {
          int naddr = _getN(addr, dim, j, Frozen);
          if (naddr != -1 && flag_[naddr] != Frozen) {
            if (flag_[naddr] == Narrow) {
              double d;
              if (order_ == 2)
                d = updatePointOrderTwo(naddr);
              else
                d = updatePointOrderOne(naddr);
              if (d) {
                heap_->set(heapptr_[naddr], fabs(d));
                distance_[naddr] = d;
              }
            } else if (flag_[naddr] == Far) {
              double d;
              if (order_ == 2)
                d = updatePointOrderTwo(naddr);
              else
                d = updatePointOrderOne(naddr);
              if (d) {
                distance_[naddr] = d;
                flag_[naddr] = Narrow;
                heapptr_[naddr] = heap_->push(naddr, fabs(d));
              }
            }
          }
          //==========================================================
          // update the far point in the second order stencil
          // "jump" over a Frozen point if needed
          if (order_ == 2) {
            int local_naddr = _getN(addr, dim, j, Mask);
            if (local_naddr != -1 && flag_[local_naddr] == Frozen) {
              int naddr2 = _getN(addr, dim, j * 2, Frozen);
              if (naddr2 != -1 && flag_[naddr2] == Narrow) {
                double d = updatePointOrderTwo(naddr2);
                if (d) {
                  heap_->set(heapptr_[naddr2], fabs(d));
                  distance_[naddr2] = d;
                }
              }
            }
          }
          //==========================================================
        } // for each direction
      } // for each dimension
    }
    batches_++;
  } // main loop of Fast Marching Method

  complete_ = true;
  return true;
}

baseMarcher::Snapshot baseMarcher::checkpoint() const {
  Snapshot state;
  state.distance.assign(distance_, distance_ + size_);
  state.flags.assign(flag_, flag_ + size_);
  state.heapptr.assign(heapptr_, heapptr_ + size_);
  state.frontier = heap_->snapshot();
  state.batches = batches_;
  state.complete = complete_;
  state.target_T = target_T_;
  return state;
}

void baseMarcher::restore(const Snapshot &state) {
  std::copy(state.distance.begin(), state.distance.end(), distance_);
  std::copy(state.flags.begin(), state.flags.end(), flag_);
  std::copy(state.heapptr.begin(), state.heapptr.end(), heapptr_);
  heap_->restore(state.frontier);
  batches_ = state.batches;
  complete_ = state.complete;
  target_T_ = state.target_T;
}

baseMarcher::Initialization baseMarcher::initialization() const {
  Initialization state;
  for (int i = 0; i < size_; i++)
    if (flag_[i] == Frozen || flag_[i] == Narrow)
      state.active.push_back({i, flag_[i], distance_[i], heapptr_[i]});
  state.frontier = heap_->snapshot();
  return state;
}

bool baseMarcher::initializationMayRead(int address) const {
  if (periodic_)
    return true;
  // Ignore masks: a later edit may expose a previously masked contour.
  const auto potential_frozen = [this](int i) {
    if (phi_[i] == 0.0)
      return true;
    for (int dim = 0; dim < dim_; dim++) {
      int coord = (i / shift_[dim]) % shape_[dim];
      // Match the initializer's product test, including underflow to zero.
      if (coord > 0 && phi_[i] * phi_[i - shift_[dim]] < 0.0)
        return true;
      if (coord + 1 < shape_[dim] && phi_[i] * phi_[i + shift_[dim]] < 0.0)
        return true;
    }
    return false;
  };
  if (potential_frozen(address))
    return true;
  // Initial Narrow cells are axial neighbors of potential Frozen cells.
  // Second-order values only use Frozen operands, already in that support.
  for (int dim = 0; dim < dim_; dim++) {
    int coord = (address / shift_[dim]) % shape_[dim];
    if (coord > 0 && potential_frozen(address - shift_[dim]))
      return true;
    if (coord + 1 < shape_[dim] && potential_frozen(address + shift_[dim]))
      return true;
  }
  return false;
}

bool baseMarcher::Initialization::operator==(const Initialization &other) const {
  return active == other.active &&
         frontier.distance == other.frontier.distance &&
         frontier.address == other.frontier.address &&
         frontier.backPointer == other.frontier.backPointer &&
         frontier.heap == other.frontier.heap;
}

std::size_t baseMarcher::Initialization::arrayBytes() const {
  return active.size() * sizeof(InitialPoint) +
         frontier.distance.size() * sizeof(double) +
         (frontier.address.size() + frontier.backPointer.size() + frontier.heap.size()) * sizeof(int);
}

std::size_t baseMarcher::arrayBytes() const {
  return static_cast<std::size_t>(size_) * sizeof(int) + heap_->arrayBytes();
}

std::size_t baseMarcher::checkpointBytes() const {
  return static_cast<std::size_t>(size_) *
         (sizeof(double) + sizeof(long long) + sizeof(int)) + heap_->snapshotBytes();
}

} // extern "C"

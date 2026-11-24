// base_marcher.h
#pragma once
#include "heap.h"
const unsigned int MaximumDimension = 12;
const char Far = 0;
const char Narrow = 1;
const char Frozen = 2;
const char Mask = 3;

#include <limits>
using namespace std;
#define doubleEpsilon numeric_limits<double>::epsilon()
#define maxDouble numeric_limits<double>::max()

extern "C" {

class baseMarcher {
public:
  baseMarcher(double *phi, double *dx, long long *flag, double *distance,
              int ndim, int *shape, bool self_test, int order, double narrow,
              int periodic);

  virtual ~baseMarcher();
  void march();
  int getError() const { return error_; }
  void setEarlyExit(int target_addr, int margin_cells);

  void initialize(bool reserve_grid = false);
  bool advance(int batch_limit = -1);
  struct Snapshot {
    std::vector<double> distance;
    std::vector<long long> flags;
    std::vector<int> heapptr;
    heap::Snapshot frontier;
    std::size_t batches;
    bool complete;
    double target_T;
  };
  Snapshot checkpoint() const;
  void restore(const Snapshot &state);
  struct InitialPoint {
    int address;
    long long flag;
    double distance;
    int heapptr;
    bool operator==(const InitialPoint &other) const {
      return address == other.address && flag == other.flag &&
             distance == other.distance && heapptr == other.heapptr;
    }
  };
  struct Initialization {
    std::vector<InitialPoint> active;
    heap::Snapshot frontier;
    bool operator==(const Initialization &other) const;
    std::size_t arrayBytes() const;
  };
  Initialization initialization() const;
  void setDependencies(long long *first_read) { first_read_ = first_read; }
  std::size_t batchCount() const { return batches_; }
  bool complete() const { return complete_; }
  bool isInert(int address) const {
    return (flag_[address] == Far || flag_[address] == Mask) &&
           heapptr_[address] == -1;
  }
  std::size_t arrayBytes() const;
  std::size_t checkpointBytes() const;

private:
  void initalizeNarrow();
  void _setIndex(int current, int coord[MaximumDimension]) {
    int rem = current;
    for (int i = 0; i < dim_; i++) {
      coord[i] = rem / shift_[i]; // integer assignment is like floor()
      rem -= coord[i] * shift_[i];
    }
  }

  int _getIndex(int coord[MaximumDimension]) {
    int ret = 0;
    for (int i = 0; i < dim_; i++)
      ret += coord[i] * shift_[i];
    return ret;
  }

  double narrow_;
  int order_;
  int *heapptr_; // heap back pointers
  heap *heap_;
  int shape_[MaximumDimension]; // size of each dimension
  int shift_[MaximumDimension];
  int periodic_;
  bool self_test_;
  std::size_t batches_;
  bool complete_;
  long long *first_read_;

protected:
  void recordDependency(int address) {
    if (first_read_ && first_read_[address] < 0)
      first_read_[address] = static_cast<long long>(batches_) + 1;
  }

  // derived classes must implement these functions
  virtual void initalizeFrozen() = 0;
  virtual double updatePointOrderTwo(int i) = 0;
  virtual double updatePointOrderOne(int i) = 0;

  // derived classes may implement these functions
  virtual void cleanUp() {}
  virtual void finalizePoint(int i, double phi_i) {}

  int _getN(int current, int dim, int dir, int flag) {
    // for a given point find the neighbor in the given dimension and
    // direction.
    // Return -1 if neighbor point is invalid (out of bounds)
    // Return -1 if neighbor point has flag_ value equal to the flag input
    int coord[MaximumDimension];
    _setIndex(current, coord);
    int newc = coord[dim] + dir;

    if (periodic_ & 1 << dim) {
      if (newc == -1)
        newc = shape_[dim] - 1;
      else if (newc == -2)
        newc = shape_[dim] - 2;
      else if (newc == shape_[dim])
        newc = 0;
      else if (newc == shape_[dim] + 1)
        newc = 1;
      coord[dim] = newc;
      int newa = _getIndex(coord);
      recordDependency(newa);
      return newa;
    }

    if (newc >= shape_[dim] || newc < 0)
      return -1;
    int newa = current + dir * shift_[dim];
    recordDependency(newa);
    if (flag_[newa] == flag)
      return -1;
    return newa;
  }

  double *distance_; // return value modified in place
  double *phi_;
  double *dx_;
  long long *flag_;
  int error_;
  int dim_;  // number of dimensions
  int size_; // flat size
  double idx2_[MaximumDimension];
  int early_exit_target_;       // flat index, -1 = disabled
  int early_exit_margin_cells_; // margin in cells
  double early_exit_margin_T_;  // precomputed: margin_cells * min(dx_)
  double target_T_; // T at which target was frozen, maxDouble = not yet
};

} // extern "C"

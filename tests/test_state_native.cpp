// Run with address and undefined-behavior sanitizers.
#include "travel_time_marcher.h"

#include <cassert>
#include <random>
#include <stdexcept>
#include <vector>

int main() {
  heap small(2, true);
  int address;
  double value;
  small.push(0, 0.0);
  small.pop(&address, &value);
  auto empty = small.snapshot();
  small.push(1, 1.0);
  small.pop(&address, &value);
  bool full = false;
  try {
    small.push(2, 2.0);
  } catch (const std::runtime_error &) {
    full = true;
  }
  assert(full);
  small.restore(empty);
  small.push(1, 2.0);
  small.pop(&address, &value);
  assert(address == 1 && value == 2.0);

  int shape[2] = {29, 31};
  int size = shape[0] * shape[1];
  double dx[2] = {0.1, 0.2};
  std::mt19937 random(19);
  std::uniform_real_distribution<double> speed_sample(0.2, 2.0);
  for (int run = 0; run < 20; run++) {
    std::vector<double> phi(size, 1.0), speed(size), distance(size, 0.0), reference(size, 0.0);
    std::vector<long long> flags(size, Far);
    int source = 14 * shape[1] + 15;
    phi[source] = -1.0;
    for (int i = 0; i < size; i++) {
      speed[i] = speed_sample(random);
      if (random() % 20 == 0 && i != source)
        flags[i] = Mask;
    }
    auto full_flags = flags;
    travelTimeMarcher full_march(phi.data(), dx, full_flags.data(), reference.data(),
                                  2, shape, true, 2, speed.data(), 0.0, 0);
    full_march.march();
    assert(full_march.getError() == 0);
    travelTimeMarcher state(phi.data(), dx, flags.data(), distance.data(),
                            2, shape, true, 2, speed.data(), 0.0, 0);
    state.initialize(true);
    auto initial = state.checkpoint();
    state.advance(17);
    auto middle = state.checkpoint();
    state.advance();
    for (const auto *saved : {&middle, &initial}) {
      state.restore(*saved);
      while (!state.advance(13)) {}
      assert(flags == full_flags);
      for (int i = 0; i < size; i++)
        if (flags[i] == Frozen)
          assert(distance[i] == reference[i]);
    }
  }
}

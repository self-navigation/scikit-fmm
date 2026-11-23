// Run with address and undefined-behavior sanitizers.
#include "travel_time_marcher.h"

#include <cassert>
#include <random>
#include <stdexcept>
#include <vector>

static void test_dependencies() {
  int shape[1] = {17};
  double dx[1] = {1.0};
  std::vector<double> phi(17, 1.0), speed(17, 1.0), distance(17, 0.0);
  std::vector<long long> flags(17, Far), first_read(17, -1);
  phi[0] = 0.0;
  flags[10] = Mask;
  travelTimeMarcher state(phi.data(), dx, flags.data(), distance.data(),
                          1, shape, true, 2, speed.data(), 0.0, 0);
  state.initialize(true);
  assert(state.getError() == 0);
  auto signature = state.initialization();
  auto initial = state.checkpoint();
  state.setDependencies(first_read.data());
  assert(!state.advance(1));
  assert(first_read[2] == 1);
  auto middle = state.checkpoint();
  assert(state.advance());
  assert(first_read[10] > 0);
  assert(first_read[15] == -1);
  for (long long event : first_read)
    assert(event == -1 ||
           (event > 0 && static_cast<std::size_t>(event) <= state.batchCount()));
  auto history = first_read;
  auto complete_flags = flags;
  auto complete_distance = distance;
  for (const auto *saved : {&initial, &middle}) {
    state.restore(*saved);
    assert(first_read == history);
    assert(state.advance());
    assert(first_read == history);
    assert(flags == complete_flags);
    assert(distance == complete_distance);
  }

  std::vector<double> changed_speed = speed, changed_distance(17, 0.0);
  std::vector<long long> changed_flags(17, Far);
  changed_speed[15] = 3.0;
  changed_flags[10] = Mask;
  changed_flags[15] = Mask;
  travelTimeMarcher inert_change(phi.data(), dx, changed_flags.data(),
                                 changed_distance.data(), 1, shape, true, 2,
                                 changed_speed.data(), 0.0, 0);
  inert_change.initialize(true);
  assert(inert_change.getError() == 0);
  assert(signature == inert_change.initialization());

  changed_speed[1] = 2.0;
  std::vector<double> narrow_distance(17, 0.0);
  std::vector<long long> narrow_flags(17, Far);
  narrow_flags[10] = Mask;
  travelTimeMarcher narrow_change(phi.data(), dx, narrow_flags.data(),
                                  narrow_distance.data(), 1, shape, true, 2,
                                  changed_speed.data(), 0.0, 0);
  narrow_change.initialize(true);
  assert(narrow_change.getError() == 0);
  assert(!(signature == narrow_change.initialization()));
}

int main() {
  test_dependencies();
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
    std::vector<long long> flags(size, Far), first_read(size, -1);
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
    state.setDependencies(first_read.data());
    auto initial = state.checkpoint();
    state.advance(17);
    auto middle = state.checkpoint();
    state.advance();
    auto history = first_read;
    for (const auto *saved : {&middle, &initial}) {
      state.restore(*saved);
      while (!state.advance(13)) {}
      assert(first_read == history);
      assert(flags == full_flags);
      for (int i = 0; i < size; i++)
        if (flags[i] == Frozen)
          assert(distance[i] == reference[i]);
    }
  }
}

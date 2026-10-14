#pragma once

#include <cuda_runtime.h>
#include "timing_lanes.h"

template <typename T>
void timingLanesForwardLauncher(TimingLaneInputs<T>, TimingLaneInputs<T>,
    const int64_t*, const int64_t*, TimingLaneOutputs<T>, T*, T*, T*,
    int, bool, T, cudaStream_t);

template <typename T>
void timingLanesBackwardLauncher(const int64_t*, const T*, const T*, const T*,
    TimingLaneInputs<T>, TimingLaneOutputs<T>, TimingLaneOutputs<T>,
    int, int, bool, cudaStream_t);

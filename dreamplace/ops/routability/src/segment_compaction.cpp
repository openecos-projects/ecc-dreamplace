#include "utility/src/torch.h"
#include "utility/src/utils.h"

#include <omp.h>

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int compactSegmentForwardLauncher(const T* seg1_llx, const T* seg1_lly,
                                  const T* seg2_llx, const T* seg2_lly,
                                  const long* seg1_indices,
                                  const long* seg2_indices,
                                  int seg1_count, int seg2_count,
                                  int num_threads, T* segment_llx,
                                  T* segment_lly) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < seg1_count; ++i) {
    const long src = seg1_indices[i];
    segment_llx[i] = seg1_llx[src];
    segment_lly[i] = seg1_lly[src];
  }

#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < seg2_count; ++i) {
    const long src = seg2_indices[i];
    const int dst = seg1_count + i;
    segment_llx[dst] = seg2_llx[src];
    segment_lly[dst] = seg2_lly[src];
  }

  return 0;
}

template <typename T>
int compactSegmentBackwardLauncher(const T* grad_segment_llx,
                                   const T* grad_segment_lly,
                                   const long* seg1_indices,
                                   const long* seg2_indices,
                                   int seg1_count, int seg2_count,
                                   int num_threads, T* grad_seg1_llx,
                                   T* grad_seg1_lly, T* grad_seg2_llx,
                                   T* grad_seg2_lly) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < seg1_count; ++i) {
    const long dst = seg1_indices[i];
    grad_seg1_llx[dst] = grad_segment_llx[i];
    grad_seg1_lly[dst] = grad_segment_lly[i];
  }

#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < seg2_count; ++i) {
    const long dst = seg2_indices[i];
    const int src = seg1_count + i;
    grad_seg2_llx[dst] = grad_segment_llx[src];
    grad_seg2_lly[dst] = grad_segment_lly[src];
  }

  return 0;
}

std::vector<at::Tensor> segment_compaction_forward(
    at::Tensor seg1_llx, at::Tensor seg1_lly, at::Tensor seg2_llx,
    at::Tensor seg2_lly, at::Tensor seg1_indices, at::Tensor seg2_indices) {
  CHECK_FLAT_CPU(seg1_llx);
  CHECK_CONTIGUOUS(seg1_llx);
  CHECK_FLAT_CPU(seg1_lly);
  CHECK_CONTIGUOUS(seg1_lly);
  CHECK_FLAT_CPU(seg2_llx);
  CHECK_CONTIGUOUS(seg2_llx);
  CHECK_FLAT_CPU(seg2_lly);
  CHECK_CONTIGUOUS(seg2_lly);
  CHECK_FLAT_CPU(seg1_indices);
  CHECK_CONTIGUOUS(seg1_indices);
  CHECK_FLAT_CPU(seg2_indices);
  CHECK_CONTIGUOUS(seg2_indices);
  AT_ASSERTM(seg1_indices.scalar_type() == at::ScalarType::Long,
             "seg1_indices must be int64");
  AT_ASSERTM(seg2_indices.scalar_type() == at::ScalarType::Long,
             "seg2_indices must be int64");

  const int seg1_count = seg1_indices.numel();
  const int seg2_count = seg2_indices.numel();
  auto segment_llx = at::empty({seg1_count + seg2_count}, seg1_llx.options());
  auto segment_lly = at::empty({seg1_count + seg2_count}, seg1_lly.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      seg1_llx, "compactSegmentForwardLauncher", [&] {
        compactSegmentForwardLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(seg1_llx, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(seg1_lly, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(seg2_llx, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(seg2_lly, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(seg1_indices, long),
            DREAMPLACE_TENSOR_DATA_PTR(seg2_indices, long), seg1_count,
            seg2_count, at::get_num_threads(),
            DREAMPLACE_TENSOR_DATA_PTR(segment_llx, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(segment_lly, scalar_t));
      });

  return {segment_llx, segment_lly};
}

std::vector<at::Tensor> segment_compaction_backward(
    at::Tensor grad_segment_llx, at::Tensor grad_segment_lly,
    at::Tensor seg1_indices, at::Tensor seg2_indices, int seg1_numel,
    int seg2_numel) {
  CHECK_FLAT_CPU(grad_segment_llx);
  CHECK_CONTIGUOUS(grad_segment_llx);
  CHECK_FLAT_CPU(grad_segment_lly);
  CHECK_CONTIGUOUS(grad_segment_lly);
  CHECK_FLAT_CPU(seg1_indices);
  CHECK_CONTIGUOUS(seg1_indices);
  CHECK_FLAT_CPU(seg2_indices);
  CHECK_CONTIGUOUS(seg2_indices);
  AT_ASSERTM(seg1_indices.scalar_type() == at::ScalarType::Long,
             "seg1_indices must be int64");
  AT_ASSERTM(seg2_indices.scalar_type() == at::ScalarType::Long,
             "seg2_indices must be int64");

  auto grad_seg1_llx = at::zeros({seg1_numel}, grad_segment_llx.options());
  auto grad_seg1_lly = at::zeros({seg1_numel}, grad_segment_lly.options());
  auto grad_seg2_llx = at::zeros({seg2_numel}, grad_segment_llx.options());
  auto grad_seg2_lly = at::zeros({seg2_numel}, grad_segment_lly.options());
  const int seg1_count = seg1_indices.numel();
  const int seg2_count = seg2_indices.numel();

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      grad_segment_llx, "compactSegmentBackwardLauncher", [&] {
        compactSegmentBackwardLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(grad_segment_llx, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_segment_lly, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(seg1_indices, long),
            DREAMPLACE_TENSOR_DATA_PTR(seg2_indices, long), seg1_count,
            seg2_count, at::get_num_threads(),
            DREAMPLACE_TENSOR_DATA_PTR(grad_seg1_llx, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_seg1_lly, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_seg2_llx, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_seg2_lly, scalar_t));
      });

  return {grad_seg1_llx, grad_seg1_lly, grad_seg2_llx, grad_seg2_lly};
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::segment_compaction_forward,
        "Segment compaction forward");
  m.def("backward", &DREAMPLACE_NAMESPACE::segment_compaction_backward,
        "Segment compaction backward");
}

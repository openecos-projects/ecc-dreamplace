#include "openroad_diff_guided_batch_bridge.h"

#include "openroad_place_io_bridge_internal.h"

namespace dreamplace {
namespace placeio_openroad {

py::dict OpenRoadPlaceIOBridge::queryDiffGuidedBatchTimingSamples(
    const py::dict& sample_request)
{
  return internal::bridgeImplQueryDiffGuidedBatchTimingSamples(impl_, sample_request);
}

py::dict OpenRoadPlaceIOBridge::queryDiffGuidedBatchTimingMetrics()
{
  return internal::bridgeImplQueryDiffGuidedBatchTimingMetrics(impl_);
}

py::dict OpenRoadPlaceIOBridge::queryDiffGuidedBatchDynamicConflictSignature(
    const py::dict& config)
{
  return internal::bridgeImplQueryDiffGuidedBatchDynamicConflictSignature(impl_, config);
}

py::dict OpenRoadPlaceIOBridge::evaluateDiffGuidedBatchActions(
    const std::vector<py::dict>& actions,
    const py::dict& config)
{
  return internal::bridgeImplEvaluateDiffGuidedBatchActions(impl_, actions, config);
}

py::dict OpenRoadPlaceIOBridge::applyDiffGuidedBatchTransaction(
    const std::vector<py::dict>& action_results,
    const py::dict& config)
{
  return internal::bridgeImplApplyDiffGuidedBatchTransaction(impl_, action_results, config);
}

py::dict OpenRoadPlaceIOBridge::runDiffGuidedBatchLoop(
    const std::vector<py::dict>& seed_queue,
    const py::dict& conflict_precompute,
    const py::dict& config)
{
  return internal::bridgeImplRunDiffGuidedBatchLoop(
      impl_,
      seed_queue,
      conflict_precompute,
      config);
}

py::dict OpenRoadPlaceIOBridge::runDiffGuidedBatchLoopCompact(
    const py::dict& action_buffers,
    const py::dict& conflict_precompute,
    const py::dict& config)
{
  return internal::bridgeImplRunDiffGuidedBatchLoopCompact(
      impl_,
      action_buffers,
      conflict_precompute,
      config);
}

}  // namespace placeio_openroad
}  // namespace dreamplace

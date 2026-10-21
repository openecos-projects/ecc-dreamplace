#include "openroad_place_io_impl_bridge.h"

namespace dreamplace {
namespace placeio_openroad {
namespace internal {

impl::OpenRoadPlaceIOBridgeImpl* checkedBridgeImpl(const std::shared_ptr<void>& impl)
{
  auto* bridge = static_cast<impl::OpenRoadPlaceIOBridgeImpl*>(impl.get());
  if (bridge == nullptr) {
    throw std::runtime_error("OpenROAD PlaceIO bridge implementation is null");
  }
  return bridge;
}

std::shared_ptr<void> makeBridgeImpl(const std::vector<std::string>& lef_files,
                                     const std::string& def_file,
                                     const std::vector<std::string>& liberty_files,
                                     const std::string& sdc_file,
                                     const std::vector<std::string>& vt_suffixes,
                                     int thread_count)
{
  return std::make_shared<impl::OpenRoadPlaceIOBridgeImpl>(
      lef_files,
      def_file,
      liberty_files,
      sdc_file,
      vt_suffixes,
      thread_count);
}

odb::dbBlock* bridgeImplBlock(const std::shared_ptr<void>& impl)
{
  return checkedBridgeImpl(impl)->block();
}

ord::Design* bridgeImplDesign(const std::shared_ptr<void>& impl)
{
  return checkedBridgeImpl(impl)->design();
}

int bridgeImplLefUnit(const std::shared_ptr<void>& impl)
{
  return checkedBridgeImpl(impl)->lefUnit();
}

int bridgeImplDefUnit(const std::shared_ptr<void>& impl)
{
  return checkedBridgeImpl(impl)->defUnit();
}

bool bridgeImplHasTimingInputs(const std::shared_ptr<void>& impl)
{
  return checkedBridgeImpl(impl)->hasTimingInputs();
}

PyPlaceDB bridgeImplExportPyDBView(const std::shared_ptr<void>& impl)
{
  return checkedBridgeImpl(impl)->exportPyDBView();
}

PyPlaceDB bridgeImplSyncFromOpenRoad(const std::shared_ptr<void>& impl)
{
  return checkedBridgeImpl(impl)->syncFromOpenRoad();
}

std::string bridgeImplEvalTclString(const std::shared_ptr<void>& impl,
                                    const std::string& cmd)
{
  return checkedBridgeImpl(impl)->evalTclString(cmd);
}

py::dict bridgeImplRefreshTiming(const std::shared_ptr<void>& impl)
{
  return checkedBridgeImpl(impl)->refreshTiming();
}

std::string bridgeImplRunBufferInsertion(const std::shared_ptr<void>& impl,
                                         const std::string& cmd)
{
  return checkedBridgeImpl(impl)->runBufferInsertion(cmd);
}

py::dict bridgeImplRunOneNetBuffer(const std::shared_ptr<void>& impl,
                                   const std::string& net_name,
                                   const py::dict& config)
{
  return checkedBridgeImpl(impl)->runOneNetBuffer(net_name, config);
}

py::dict bridgeImplRunCoordinateBufferInsert(const std::shared_ptr<void>& impl,
                                             const py::dict& action,
                                             const py::dict& config)
{
  return checkedBridgeImpl(impl)->runCoordinateBufferInsert(action, config);
}

void bridgeImplSetNodeOrient(const std::shared_ptr<void>& impl,
                             int node_id,
                             int orient_value)
{
  checkedBridgeImpl(impl)->setNodeOrient(node_id, orient_value);
}

void bridgeImplSyncToOpenRoad(const std::shared_ptr<void>& impl,
                              const py::object& node_x,
                              const py::object& node_y)
{
  checkedBridgeImpl(impl)->syncToOpenRoad(node_x, node_y);
}

py::dict bridgeImplApplySizing(const std::shared_ptr<void>& impl,
                               const py::object& inst_cell_ids,
                               const py::object& cell_master_names)
{
  return checkedBridgeImpl(impl)->applySizing(inst_cell_ids, cell_master_names);
}

py::dict bridgeImplQueryDiffGuidedBatchTimingSamples(const std::shared_ptr<void>& impl,
                                                     const py::dict& sample_request)
{
  return checkedBridgeImpl(impl)->queryDiffGuidedBatchTimingSamples(sample_request);
}

py::dict bridgeImplQueryDiffGuidedBatchTimingMetrics(const std::shared_ptr<void>& impl)
{
  return checkedBridgeImpl(impl)->queryDiffGuidedBatchTimingMetrics();
}

py::dict bridgeImplQueryDiffGuidedBatchDynamicConflictSignature(
    const std::shared_ptr<void>& impl,
    const py::dict& config)
{
  return checkedBridgeImpl(impl)->queryDiffGuidedBatchDynamicConflictSignature(config);
}

py::dict bridgeImplEvaluateDiffGuidedBatchActions(const std::shared_ptr<void>& impl,
                                                  const std::vector<py::dict>& actions,
                                                  const py::dict& config)
{
  return checkedBridgeImpl(impl)->evaluateDiffGuidedBatchActions(actions, config);
}

py::dict bridgeImplApplyDiffGuidedBatchTransaction(
    const std::shared_ptr<void>& impl,
    const std::vector<py::dict>& action_results,
    const py::dict& config)
{
  return checkedBridgeImpl(impl)->applyDiffGuidedBatchTransaction(action_results, config);
}

py::dict bridgeImplRunDiffGuidedBatchLoop(const std::shared_ptr<void>& impl,
                                          const std::vector<py::dict>& seed_queue,
                                          const py::dict& conflict_precompute,
                                          const py::dict& config)
{
  return checkedBridgeImpl(impl)->runDiffGuidedBatchLoop(
      seed_queue,
      conflict_precompute,
      config);
}

py::dict bridgeImplRunDiffGuidedBatchLoopCompact(const std::shared_ptr<void>& impl,
                                                 const py::dict& action_buffers,
                                                 const py::dict& conflict_precompute,
                                                 const py::dict& config)
{
  return checkedBridgeImpl(impl)->runDiffGuidedBatchLoopCompact(
      action_buffers,
      conflict_precompute,
      config);
}

void bridgeImplWriteDef(const std::shared_ptr<void>& impl, const std::string& filename)
{
  checkedBridgeImpl(impl)->writeDef(filename);
}

}  // namespace internal
}  // namespace placeio_openroad
}  // namespace dreamplace

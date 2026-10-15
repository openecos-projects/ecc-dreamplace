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

namespace {

using dreamplace::placeio_openroad::OpenRoadPlaceIOBridge;
using dreamplace::placeio_openroad::PyPlaceDB;

std::vector<std::string> parseFlagValues(const py::list& args, const std::string& flag)
{
  std::vector<std::string> values;
  const auto arg_count = py::len(args);
  for (py::ssize_t i = 0; i < arg_count; ++i) {
    if (py::cast<std::string>(args[i]) == flag && i + 1 < arg_count) {
      values.push_back(py::cast<std::string>(args[i + 1]));
    }
  }
  return values;
}

int parseFlagInt(const py::list& args, const std::string& flag, int default_value)
{
  const auto values = parseFlagValues(args, flag);
  if (values.empty()) {
    return default_value;
  }
  try {
    return std::max(1, std::stoi(values.back()));
  } catch (const std::exception&) {
    return default_value;
  }
}

using BridgePtr = std::shared_ptr<OpenRoadPlaceIOBridge>;

BridgePtr forward(const py::list& args)
{
  const auto lef_files = parseFlagValues(args, "--lef_input");
  const auto def_files = parseFlagValues(args, "--def_input");
  const auto liberty_files = parseFlagValues(args, "--lib_input");
  const auto sdc_files = parseFlagValues(args, "--sdc_input");
  auto vt_suffixes = parseFlagValues(args, "--vt_suffix");
  const int thread_count = parseFlagInt(args, "--num_threads", 8);
  if (def_files.empty()) {
    throw std::runtime_error("placeio_openroad requires exactly one DEF input");
  }
  if (sdc_files.size() > 1) {
    throw std::runtime_error("placeio_openroad accepts at most one SDC input");
  }
  return std::make_shared<OpenRoadPlaceIOBridge>(
      lef_files,
      def_files.front(),
      liberty_files,
      sdc_files.empty() ? std::string() : sdc_files.front(),
      vt_suffixes,
      thread_count);
}

PyPlaceDB pydb(const BridgePtr& bridge)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->exportPyDBView();
}

void apply(const BridgePtr& bridge, const py::object& node_x, const py::object& node_y)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  bridge->syncToOpenRoad(node_x, node_y);
}

py::dict apply_sizing(const BridgePtr& bridge,
                      const py::object& inst_cell_ids,
                      const py::object& cell_master_names)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->applySizing(inst_cell_ids, cell_master_names);
}

py::dict query_diff_guided_batch_timing_metrics(const BridgePtr& bridge)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->queryDiffGuidedBatchTimingMetrics();
}

py::dict query_diff_guided_batch_timing_samples(const BridgePtr& bridge,
                                                const py::dict& sample_request)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->queryDiffGuidedBatchTimingSamples(sample_request);
}

py::dict query_diff_guided_batch_dynamic_conflict_signature(const BridgePtr& bridge,
                                                            const py::dict& config)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->queryDiffGuidedBatchDynamicConflictSignature(config);
}

py::dict evaluate_diff_guided_batch_actions(const BridgePtr& bridge,
                                            const std::vector<py::dict>& actions,
                                            const py::dict& config)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->evaluateDiffGuidedBatchActions(actions, config);
}

py::dict apply_diff_guided_batch_transaction(const BridgePtr& bridge,
                                             const std::vector<py::dict>& action_results,
                                             const py::dict& config)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->applyDiffGuidedBatchTransaction(action_results, config);
}

py::dict run_diff_guided_batch_loop(const BridgePtr& bridge,
                                    const std::vector<py::dict>& seed_queue,
                                    const py::dict& conflict_precompute,
                                    const py::dict& config)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->runDiffGuidedBatchLoop(seed_queue, conflict_precompute, config);
}

py::dict run_diff_guided_batch_loop_compact(const BridgePtr& bridge,
                                            const py::dict& action_buffers,
                                            const py::dict& conflict_precompute,
                                            const py::dict& config)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->runDiffGuidedBatchLoopCompact(action_buffers, conflict_precompute, config);
}

void write(const BridgePtr& bridge,
           const std::string& filename,
           int,
           const py::object& node_x,
           const py::object& node_y)
{
  apply(bridge, node_x, node_y);
  bridge->writeDef(filename);
}

}  // namespace

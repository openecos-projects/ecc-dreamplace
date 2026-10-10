#pragma once

#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openroad_pyplacedb_export.h"

namespace odb {
class dbBlock;
}

namespace ord {
class Design;
}

namespace py = pybind11;

namespace dreamplace {
namespace placeio_openroad {
namespace internal {

std::shared_ptr<void> makeBridgeImpl(const std::vector<std::string>& lef_files,
                                     const std::string& def_file,
                                     const std::vector<std::string>& liberty_files,
                                     const std::string& sdc_file,
                                     const std::vector<std::string>& vt_suffixes,
                                     int thread_count);
odb::dbBlock* bridgeImplBlock(const std::shared_ptr<void>& impl);
ord::Design* bridgeImplDesign(const std::shared_ptr<void>& impl);
int bridgeImplLefUnit(const std::shared_ptr<void>& impl);
int bridgeImplDefUnit(const std::shared_ptr<void>& impl);
bool bridgeImplHasTimingInputs(const std::shared_ptr<void>& impl);
PyPlaceDB bridgeImplExportPyDBView(const std::shared_ptr<void>& impl);
PyPlaceDB bridgeImplSyncFromOpenRoad(const std::shared_ptr<void>& impl);
std::string bridgeImplEvalTclString(const std::shared_ptr<void>& impl, const std::string& cmd);
py::dict bridgeImplRefreshTiming(const std::shared_ptr<void>& impl);
std::string bridgeImplRunBufferInsertion(const std::shared_ptr<void>& impl,
                                         const std::string& cmd);
py::dict bridgeImplRunOneNetBuffer(const std::shared_ptr<void>& impl,
                                   const std::string& net_name,
                                   const py::dict& config);
py::dict bridgeImplRunCoordinateBufferInsert(const std::shared_ptr<void>& impl,
                                             const py::dict& action,
                                             const py::dict& config);
void bridgeImplSetNodeOrient(const std::shared_ptr<void>& impl, int node_id, int orient_value);
void bridgeImplSyncToOpenRoad(const std::shared_ptr<void>& impl,
                              const py::object& node_x,
                              const py::object& node_y);
py::dict bridgeImplApplySizing(const std::shared_ptr<void>& impl,
                               const py::object& inst_cell_ids,
                               const py::object& cell_master_names);
py::dict bridgeImplQueryDiffGuidedBatchTimingSamples(const std::shared_ptr<void>& impl,
                                                     const py::dict& sample_request);
py::dict bridgeImplQueryDiffGuidedBatchTimingMetrics(const std::shared_ptr<void>& impl);
py::dict bridgeImplQueryDiffGuidedBatchDynamicConflictSignature(const std::shared_ptr<void>& impl,
                                                               const py::dict& config);
py::dict bridgeImplEvaluateDiffGuidedBatchActions(const std::shared_ptr<void>& impl,
                                                  const std::vector<py::dict>& actions,
                                                  const py::dict& config);
py::dict bridgeImplApplyDiffGuidedBatchTransaction(
    const std::shared_ptr<void>& impl,
    const std::vector<py::dict>& action_results,
    const py::dict& config);
py::dict bridgeImplRunDiffGuidedBatchLoop(const std::shared_ptr<void>& impl,
                                          const std::vector<py::dict>& seed_queue,
                                          const py::dict& conflict_precompute,
                                          const py::dict& config);
py::dict bridgeImplRunDiffGuidedBatchLoopCompact(const std::shared_ptr<void>& impl,
                                                 const py::dict& action_buffers,
                                                 const py::dict& conflict_precompute,
                                                 const py::dict& config);
void bridgeImplWriteDef(const std::shared_ptr<void>& impl, const std::string& filename);

}  // namespace internal
}  // namespace placeio_openroad
}  // namespace dreamplace

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

class OpenRoadPlaceIOBridge
{
 public:
  OpenRoadPlaceIOBridge(const std::vector<std::string>& lef_files,
                        const std::string& def_file,
                        const std::vector<std::string>& liberty_files,
                        const std::string& sdc_file,
                        const std::vector<std::string>& vt_suffixes,
                        int thread_count);

  odb::dbBlock* block() const;
  ord::Design* design() const;
  int lefUnit() const;
  int defUnit() const;
  bool hasTimingInputs() const;
  PyPlaceDB exportPyDBView();
  PyPlaceDB syncFromOpenRoad();
  std::string evalTclString(const std::string& cmd);
  py::dict refreshTiming();
  std::string runBufferInsertion(const std::string& cmd);
  py::dict runOneNetBuffer(const std::string& net_name, const py::dict& config);
  py::dict runCoordinateBufferInsert(const py::dict& action, const py::dict& config);
  void setNodeOrient(int node_id, int orient_value);
  void syncToOpenRoad(const py::object& node_x, const py::object& node_y);
  py::dict applySizing(const py::object& inst_cell_ids, const py::object& cell_master_names);
  py::dict queryDiffGuidedBatchTimingSamples(const py::dict& sample_request);
  py::dict queryDiffGuidedBatchTimingMetrics();
  py::dict queryDiffGuidedBatchDynamicConflictSignature(const py::dict& config);
  py::dict evaluateDiffGuidedBatchActions(const std::vector<py::dict>& actions,
                                          const py::dict& config);
  py::dict applyDiffGuidedBatchTransaction(const std::vector<py::dict>& action_results,
                                           const py::dict& config);
  py::dict runDiffGuidedBatchLoop(const std::vector<py::dict>& seed_queue,
                                  const py::dict& conflict_precompute,
                                  const py::dict& config);
  py::dict runDiffGuidedBatchLoopCompact(const py::dict& action_buffers,
                                         const py::dict& conflict_precompute,
                                         const py::dict& config);
  void writeDef(const std::string& filename) const;

 private:
  std::shared_ptr<void> impl_;
};

}  // namespace placeio_openroad
}  // namespace dreamplace

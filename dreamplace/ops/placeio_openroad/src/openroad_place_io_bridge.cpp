#include "openroad_place_io_bridge.h"

#include "openroad_place_io_bridge_internal.h"

namespace dreamplace {
namespace placeio_openroad {

OpenRoadPlaceIOBridge::OpenRoadPlaceIOBridge(
    const std::vector<std::string>& lef_files,
    const std::string& def_file,
    const std::vector<std::string>& liberty_files,
    const std::string& sdc_file,
    const std::vector<std::string>& vt_suffixes,
    int thread_count)
    : impl_(internal::makeBridgeImpl(
          lef_files,
          def_file,
          liberty_files,
          sdc_file,
          vt_suffixes,
          thread_count))
{
}

odb::dbBlock* OpenRoadPlaceIOBridge::block() const
{
  return internal::bridgeImplBlock(impl_);
}

ord::Design* OpenRoadPlaceIOBridge::design() const
{
  return internal::bridgeImplDesign(impl_);
}

int OpenRoadPlaceIOBridge::lefUnit() const
{
  return internal::bridgeImplLefUnit(impl_);
}

int OpenRoadPlaceIOBridge::defUnit() const
{
  return internal::bridgeImplDefUnit(impl_);
}

bool OpenRoadPlaceIOBridge::hasTimingInputs() const
{
  return internal::bridgeImplHasTimingInputs(impl_);
}

PyPlaceDB OpenRoadPlaceIOBridge::exportPyDBView()
{
  return internal::bridgeImplExportPyDBView(impl_);
}

PyPlaceDB OpenRoadPlaceIOBridge::syncFromOpenRoad()
{
  return internal::bridgeImplSyncFromOpenRoad(impl_);
}

std::string OpenRoadPlaceIOBridge::evalTclString(const std::string& cmd)
{
  return internal::bridgeImplEvalTclString(impl_, cmd);
}

py::dict OpenRoadPlaceIOBridge::refreshTiming()
{
  return internal::bridgeImplRefreshTiming(impl_);
}

std::string OpenRoadPlaceIOBridge::runBufferInsertion(const std::string& cmd)
{
  return internal::bridgeImplRunBufferInsertion(impl_, cmd);
}

py::dict OpenRoadPlaceIOBridge::runOneNetBuffer(const std::string& net_name,
                                                const py::dict& config)
{
  return internal::bridgeImplRunOneNetBuffer(impl_, net_name, config);
}

py::dict OpenRoadPlaceIOBridge::runCoordinateBufferInsert(const py::dict& action,
                                                          const py::dict& config)
{
  return internal::bridgeImplRunCoordinateBufferInsert(impl_, action, config);
}

void OpenRoadPlaceIOBridge::setNodeOrient(int node_id, int orient_value)
{
  internal::bridgeImplSetNodeOrient(impl_, node_id, orient_value);
}

void OpenRoadPlaceIOBridge::syncToOpenRoad(const py::object& node_x,
                                           const py::object& node_y)
{
  internal::bridgeImplSyncToOpenRoad(impl_, node_x, node_y);
}

py::dict OpenRoadPlaceIOBridge::applySizing(const py::object& inst_cell_ids,
                                            const py::object& cell_master_names)
{
  return internal::bridgeImplApplySizing(impl_, inst_cell_ids, cell_master_names);
}

void OpenRoadPlaceIOBridge::writeDef(const std::string& filename) const
{
  internal::bridgeImplWriteDef(impl_, filename);
}

}  // namespace placeio_openroad
}  // namespace dreamplace

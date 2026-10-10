#pragma once

// OpenROAD-backed diff-guided batch bridge declarations are currently exposed
// through OpenRoadPlaceIOBridge.  This header marks the file boundary for the
// diff-guided batch facade methods; the OpenSTA mutation/query hot path remains
// inside OpenRoadPlaceIOBridgeImpl until that implementation is split safely.
#include "openroad_place_io_bridge.h"

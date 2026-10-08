#pragma once

#include <string>
#include <vector>

namespace odb {
class dbBTerm;
class dbITerm;
}

namespace dreamplace {
namespace placeio_openroad {

struct PinRecord
{
  std::string name;
  int node_id;
  int net_id;
  int offset_x;
  int offset_y;
  std::string direction;
  bool is_io{false};
};

struct NetRecord
{
  std::string name;
  std::vector<int> pin_ids;
  int driver_pin_id{-1};
};

struct PendingPinRecord
{
  PinRecord pin;
  odb::dbITerm* iterm{nullptr};
  odb::dbBTerm* bterm{nullptr};
};

}  // namespace placeio_openroad
}  // namespace dreamplace

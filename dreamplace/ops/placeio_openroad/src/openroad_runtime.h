#pragma once

#include <memory>

namespace utl {
class Logger;
}

namespace dreamplace {
namespace placeio_openroad {

class OpenRoadRuntime
{
 public:
  static OpenRoadRuntime& instance();

  utl::Logger* logger() const;

 private:
  OpenRoadRuntime();
  ~OpenRoadRuntime();
  OpenRoadRuntime(const OpenRoadRuntime&) = delete;
  OpenRoadRuntime& operator=(const OpenRoadRuntime&) = delete;

  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace placeio_openroad
}  // namespace dreamplace

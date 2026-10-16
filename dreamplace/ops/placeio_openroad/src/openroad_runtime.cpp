#include "openroad_runtime.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "ord/OpenRoad.hh"
#include "ord/ordMain.hh"
#include "utl/Logger.h"

namespace dreamplace {
namespace placeio_openroad {

class OpenRoadRuntime::Impl
{
 public:
  Impl()
  {
    args_storage_ = {"DREAMPlace", "-no_splash"};
    argv_.reserve(args_storage_.size());
    for (std::string& arg : args_storage_) {
      argv_.push_back(arg.data());
    }

    omp_set_num_threads(8);
    ord::flow_OpenROAD(static_cast<int>(argv_.size()), argv_.data());

    auto* openroad = ord::OpenRoad::openRoad();
    if (openroad == nullptr) {
      throw std::runtime_error("OpenROAD runtime initialization failed");
    }
    openroad->setThreadCount(8, false);
    logger_ = openroad->getLogger();
    if (logger_ == nullptr) {
      throw std::runtime_error("OpenROAD runtime did not provide a logger");
    }
  }

  utl::Logger* logger() const { return logger_; }

 private:
  utl::Logger* logger_{nullptr};
  std::vector<std::string> args_storage_;
  std::vector<char*> argv_;
};

OpenRoadRuntime& OpenRoadRuntime::instance()
{
  static OpenRoadRuntime runtime;
  return runtime;
}

utl::Logger* OpenRoadRuntime::logger() const
{
  return impl_->logger();
}

OpenRoadRuntime::OpenRoadRuntime() : impl_(std::make_unique<Impl>()) {}

OpenRoadRuntime::~OpenRoadRuntime() = default;

}  // namespace placeio_openroad
}  // namespace dreamplace

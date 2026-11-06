// SPDX-License-Identifier: BSD-3-Clause
#include "rebuffer_timing_policy.hh"
#include "rebuffer_electrical_observation.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace py = pybind11;
namespace rebuffer {
namespace {

double number(const py::dict& values, const char* key, double fallback)
{
  return values.contains(key) && !values[key].is_none()
             ? py::cast<double>(values[key]) : fallback;
}

bool boolean(const py::dict& values, const char* key, bool fallback)
{
  return values.contains(key) && !values[key].is_none()
             ? py::cast<bool>(values[key]) : fallback;
}

}  // namespace

bool acceptsTimingCandidate(const py::dict& current, const py::dict& candidate,
                            const py::dict& request, const py::dict& config)
{
  if (candidate.contains("status") && py::cast<std::string>(candidate["status"]) != "ok") return false;
  const double missing = std::numeric_limits<double>::quiet_NaN();
  const double wns = number(candidate, "wns_ps", missing);
  const double tns = number(candidate, "tns_ps", missing);
  const double old_wns = number(current, "wns_ps", missing);
  const double old_tns = number(current, "tns_ps", missing);
  if (!std::isfinite(wns) || !std::isfinite(tns)
      || !std::isfinite(old_wns) || !std::isfinite(old_tns)) return false;
  if (!electricalNonRegressing(current, candidate)) return false;
  constexpr double epsilon = 1.0e-9;
  if (config.contains("acceptance")
      && config["acceptance"].cast<std::string>() == "openroad_legacy") {
    const bool endpoint_phase = request.contains("legacy_phase")
        && request["legacy_phase"].cast<std::string>() == "endpoint";
    const double previous = number(request, "previous_endpoint_slack_ps", missing);
    const double endpoint = number(candidate, "endpoint_slack_ps", missing);
    return wns > old_wns + epsilon
        || (endpoint_phase && std::abs(wns - old_wns) <= epsilon
            && std::isfinite(previous) && std::isfinite(endpoint)
            && endpoint > previous + epsilon);
  }
  const double margin = std::max(0.0, number(config, "tns_regression_margin_ps", 0.0));
  // Every accepted epoch protects global WNS and the configured TNS envelope,
  // including endpoint-journal fallback and branch-focused candidates.
  if (wns < old_wns - epsilon || tns < old_tns - margin) return false;
  if (wns > old_wns + epsilon || tns > old_tns + margin) return true;
  const bool branch = boolean(config, "native_tree_branch_queue", false)
      && boolean(config, "native_tree_branch_global_acceptance", true)
      && request.contains("focus_net_id") && request.contains("root_net_id")
      && py::cast<int>(request["focus_net_id"]) != py::cast<int>(request["root_net_id"]);
  if (branch || !config.contains("acceptance")
      || py::cast<std::string>(config["acceptance"]) != "endpoint_journal"
      || !boolean(config, "allow_decreasing_slack", false)) return false;
  const double previous = number(candidate, "previous_endpoint_slack_ps",
      number(request, "previous_endpoint_slack_ps", number(current, "endpoint_slack_ps", missing)));
  const double endpoint = number(candidate, "endpoint_slack_ps", missing);
  return std::isfinite(previous) && std::isfinite(endpoint) && endpoint > previous + epsilon;
}

}  // namespace rebuffer

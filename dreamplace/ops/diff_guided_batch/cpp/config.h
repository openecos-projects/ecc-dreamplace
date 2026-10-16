#pragma once

#include <string>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dreamplace {
namespace diff_guided_batch {

struct EffectiveLocalWeight
{
  double value{0.0};
  std::string source{"explicit_local_weight"};
};

int selectorWorkerCount(const py::dict& config);
int trialWorkerCount(const py::dict& config);
std::string acceptMode(const py::dict& config);
double tnsPowerLeakageWeight(const py::dict& config);
bool localObjectiveAlignGlobal(const py::dict& config);
bool noopBaseline(const py::dict& config);
bool rejectNonpositiveTrial(const py::dict& config);
bool forceAcceptSelectedActions(const py::dict& config);
bool singleActionAudit(const py::dict& config);
int singleActionAuditLimit(const py::dict& config);
double nonpositiveTrialEps(const py::dict& config);
double localDeltaSlewPenaltyWeight(const py::dict& config);
double localDeltaCapPenaltyWeight(const py::dict& config);
double faninPenaltySensitivityPsPerPf(const py::dict& config);
std::string slackDeltaEvaluatorMode(const py::dict& config);
std::string slackDeltaWeightMode(const py::dict& config);
double npathWeightCap(const py::dict& config);
bool laneContains(const std::string& lane, const std::string& term);
EffectiveLocalWeight effectiveLocalDeltaSlewPenaltyWeight(const py::dict& config);
EffectiveLocalWeight effectiveLocalDeltaCapPenaltyWeight(const py::dict& config);
double localResidualFeedbackScale(const py::dict& config);
std::string localResidualFeedbackPolicy(const py::dict& config);
double localResidualFeedbackDecay(const py::dict& config);
double localResidualMaxGainPerAction(const py::dict& config);
double localResidualBudgetDiscount(const py::dict& config);
std::string trialMode(const py::dict& config);
int trialMiniBatchSize(const py::dict& config);

}  // namespace diff_guided_batch
}  // namespace dreamplace

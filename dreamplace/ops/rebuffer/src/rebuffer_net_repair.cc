// SPDX-License-Identifier: BSD-3-Clause
// OpenROAD-style recursive repairNet pass over one Steiner tree.

#include "rebuffer_repair.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace rebuffer {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

struct NetNode
{
  enum class Type
  {
    Wire,
    Junction,
    Load
  };

  Type type = Type::Load;
  int original_node = -1;
  int x = 0;
  int y = 0;
  int child = -1;
  std::vector<int> children;
  double cap = 0.0;
  double max_load_slew = kInf;
  int wire_length = 0;
  int fanout = 0;
  // Original terminal closure for the placeio_ecc ancestor-first commit.
  // Electrical fields above describe the current repeater input boundary.
  std::vector<int> load_pins;
};

class NetTreeRepair
{
 public:
  NetTreeRepair(const DesignView& design,
                const TimingView& timing,
                const SteinerView& steiner,
                const LibertyView& liberty,
                const CharacterizationResult& characterization,
                const RepairConfig& config,
                int net_id,
                int driver,
                int driver_cell_index,
                double max_cap,
                const std::vector<std::vector<int>>& children)
      : design_(design),
        timing_(timing),
        steiner_(steiner),
        liberty_(liberty),
        characterization_(characterization),
        config_(config),
        net_id_(net_id),
        driver_(driver),
        max_cap_(max_cap > 0.0 ? max_cap : kInf),
        children_(children)
  {
    if (config_.repair_design_max_wire_length_um > 0.0 && design_.dbu > 0) {
      const double max_length_dbu
          = config_.repair_design_max_wire_length_um * design_.dbu;
      max_wire_length_dbu_
          = max_length_dbu >= std::numeric_limits<int>::max()
                ? std::numeric_limits<int>::max()
                : static_cast<int>(std::lround(max_length_dbu));
    }
    const int lib_pin = rebufferLibPinIndex(design_, driver_);
    if (lib_pin >= 0
        && lib_pin < static_cast<int>(design_.pin_drive_resistance.size())) {
      driver_resistance_ = design_.pin_drive_resistance[lib_pin];
    }
    // The planner may already have resized this driver. Use its current
    // cell rather than resolving the master from the immutable DesignView.
    if (driver_cell_index >= 0
        && driver_cell_index < static_cast<int>(liberty_.cells.size())) {
      const double liberty_resistance
          = cellDriveResistance(liberty_.cells[driver_cell_index]);
      if (liberty_resistance > 0.0) {
        driver_resistance_ = liberty_resistance;
      }
    }
    // OpenROAD clips the wire-repair resistance to the strongest buffer so a
    // top-level port with no drive still gets a finite split point.  Junction
    // load-slew checks use the raw driver resistance, so keep both values.
    driver_wire_resistance_ = driver_resistance_;
    double strongest = kInf;
    for (const auto& buffer : characterization_.filtered_buffers) {
      if (buffer.drive_resistance > 0.0) {
        strongest = std::min(strongest, buffer.drive_resistance);
      }
    }
    if (std::isfinite(strongest)) {
      driver_wire_resistance_ = std::max(driver_wire_resistance_, strongest);
    }
    driver_slew_rise_ = timing_.pin_slew_rise[driver_];
    driver_slew_fall_ = timing_.pin_slew_fall[driver_];
  }

  bool run(std::vector<BufferAction>& actions,
           std::string& reason,
           double* root_cap_out = nullptr)
  {
    actions_ = &actions;
    const int root = buildNode(driver_);
    if (root < 0) {
      return true;
    }
    repairNode(root);
    if (root_cap_out != nullptr
        && root < static_cast<int>(nodes_.size())) {
      *root_cap_out = nodes_[root].cap;
    }
    return true;
  }

 private:
  int buildNode(int node)
  {
    if (node < 0 || node >= static_cast<int>(children_.size())) {
      return -1;
    }
    // A physical load may lie on the path to another load. OpenROAD creates
    // its load node before folding the outgoing wires into binary junctions.
    int result = -1;
    if (node < steiner_.steiner_node_base && node != driver_) {
      NetNode load;
      load.type = NetNode::Type::Load;
      load.original_node = node;
      load.x = static_cast<int>(std::lround(steiner_.node_x[node]));
      load.y = static_cast<int>(std::lround(steiner_.node_y[node]));
      load.cap = node < static_cast<int>(timing_.pin_cap.size())
                     ? timing_.pin_cap[node]
                     : 0.0;
      load.max_load_slew = node < static_cast<int>(timing_.pin_max_slew.size())
                               && timing_.pin_max_slew[node] > 0.0
                           ? timing_.pin_max_slew[node]
                           : kInf;
      load.wire_length = 0;
      load.fanout = 1;
      load.load_pins.push_back(node);
      nodes_.push_back(std::move(load));
      result = static_cast<int>(nodes_.size()) - 1;
    }
    for (const int child : children_[node]) {
      const int edge = buildEdge(node, child);
      if (edge < 0) {
        return -1;
      }
      if (result < 0) {
        result = edge;
      } else {
        NetNode junction;
        junction.type = NetNode::Type::Junction;
        junction.original_node = node;
        junction.x = static_cast<int>(std::lround(steiner_.node_x[node]));
        junction.y = static_cast<int>(std::lround(steiner_.node_y[node]));
        junction.children = {result, edge};
        nodes_.push_back(std::move(junction));
        result = static_cast<int>(nodes_.size()) - 1;
      }
    }
    return result;
  }

  int buildEdge(int parent, int child)
  {
    const int child_index = buildNode(child);
    if (child_index < 0) {
      return -1;
    }
    NetNode wire;
    wire.type = NetNode::Type::Wire;
    wire.original_node = parent;
    wire.x = static_cast<int>(std::lround(steiner_.node_x[parent]));
    wire.y = static_cast<int>(std::lround(steiner_.node_y[parent]));
    wire.child = child_index;
    nodes_.push_back(std::move(wire));
    return static_cast<int>(nodes_.size()) - 1;
  }

  void repairNode(int index)
  {
    if (index < 0 || index >= static_cast<int>(nodes_.size())) {
      return;
    }
    switch (nodes_[index].type) {
      case NetNode::Type::Wire:
        repairWire(index);
        break;
      case NetNode::Type::Junction:
        repairJunction(index);
        break;
      case NetNode::Type::Load:
        break;
    }
  }

  void repairWire(int index)
  {
    NetNode& wire = nodes_[index];
    const int child_index = wire.child;
    if (child_index < 0 || child_index >= static_cast<int>(nodes_.size())) {
      return;
    }
    repairNode(child_index);
    NetNode& child = nodes_[child_index];
    const int from_x = wire.x;
    const int from_y = wire.y;
    int to_x = child.x;
    int to_y = child.y;
    int length_dbu = std::abs(from_x - to_x) + std::abs(from_y - to_y);
    double ref_cap = child.cap;
    double wire_res = 0.0;
    double wire_cap = 0.0;
    updateLength(length_dbu, wire_res, wire_cap);
    double load_cap = wire_cap + ref_cap;
    double max_load_slew = child.max_load_slew;
    int wire_length_ref = child.wire_length;
    int fanout = child.fanout;
    double load_slew
        = estimateSlew(wire_length_ref, wire_res, wire_cap, ref_cap);
    int target_buffer_cell = rebufferChooseBufferCell(
        liberty_,
        characterization_,
        driver_slew_rise_,
        driver_slew_fall_,
        load_cap,
        config_.repair_design_target_cell_math,
        config_.repair_design_target_cell_selection,
        -1,
        config_.sizing_area_limit,
        config_.sizing_leakage_limit);

    // OpenROAD backs up the segment inserting repeaters until the remaining
    // upstream wire meets max_cap / max_load_slew / max_wire_length.  Each
    // repeater becomes the new driver reference for the next iteration.
    int zero_progress_iters = 0;
    int wire_iters = 0;
    const int max_wire_iters
        = config_.tree_max_buffers > 0 ? config_.tree_max_buffers : 1024;
    while (length_dbu > 0 && wire_iters < max_wire_iters) {
      ++wire_iters;
      const int total_wire_length = wire_length_ref + length_dbu;
      const double max_load_slew_margined
          = max_load_slew * (1.0 - config_.slew_margin);
      const bool wire_length_violation
          = max_wire_length_dbu_ > 0
            && total_wire_length > max_wire_length_dbu_;
      const bool cap_violation
          = std::isfinite(max_cap_) && max_cap_ > 0.0 && load_cap > max_cap_;
      const bool negative_limit = max_load_slew < 0.0;
      const bool slew_violation
          = std::isfinite(max_load_slew_margined)
            && load_slew > max_load_slew_margined;
      if (!negative_limit && !wire_length_violation && !cap_violation
          && !slew_violation) {
        break;
      }
      bool resize_repeater = true;
      int split_length_dbu = length_dbu;
      if (negative_limit) {
        split_length_dbu = 0;
      }
      if (wire_length_violation) {
        split_length_dbu = std::min(
            split_length_dbu,
            std::max(max_wire_length_dbu_ - wire_length_ref, 0));
      }
      if (cap_violation && config_.c_unit > 0.0) {
        const double cap_split_um
            = std::max((max_cap_ - ref_cap) / config_.c_unit, 0.0);
        split_length_dbu = std::min(
            split_length_dbu,
            static_cast<int>(std::lround(cap_split_um * design_.dbu)));
      }
      if (slew_violation) {
        const double slew_split_um = solveSlewLength(
            wire_length_ref, ref_cap, max_load_slew_margined, target_buffer_cell);
        if (slew_split_um >= 0.0) {
          split_length_dbu = std::min(
              split_length_dbu,
              static_cast<int>(std::lround(slew_split_um * design_.dbu)));
          resize_repeater = false;
        } else {
          split_length_dbu = 0;
          resize_repeater = true;
        }
      }
      if (split_length_dbu < 0) {
        split_length_dbu = 0;
      }
      // Once the downstream load is already isolated by a repeater, if the
      // upstream driver still cannot drive any meaningful wire length, the
      // isolating repeater belongs at the upstream driver end rather than
      // stepping 1 DBU at a time from the load end.
      if (wire_iters > 1 && split_length_dbu <= std::max(1, design_.dbu)) {
        split_length_dbu = length_dbu;
      }
      const int buf_dist_dbu
          = split_length_dbu >= length_dbu
                ? length_dbu
                : static_cast<int>(std::lround(split_length_dbu * 0.95));
      if (buf_dist_dbu < 0) {
        break;
      }
      int placement_dist_dbu = buf_dist_dbu;
      if (placement_dist_dbu == 0 && length_dbu > 0) {
        // Keep the zero-progress electrical state used by the recursive
        // model, but do not place the physical instance exactly on the
        // downstream junction that was just repaired.  A one-micron DBU
        // displacement (or the whole remaining edge when shorter) is the
        // smallest stable placement separation before legal-site recovery.
        placement_dist_dbu
            = std::min(length_dbu, std::max(1, design_.dbu));
      }
      const double buffer_load_cap
          = (static_cast<double>(buf_dist_dbu) / design_.dbu)
                * config_.c_unit
            + ref_cap;
      const auto selection = rebufferChooseRepeaterCell(
          liberty_,
          characterization_,
          driver_slew_rise_,
          driver_slew_fall_,
          load_cap,
          buffer_load_cap,
          config_,
          resize_repeater);
      int buffer_cell = selection.cell_index;
      if (buffer_cell < 0) {
        break;
      }
      const double prev_ref_cap = ref_cap;
      const double fraction
          = static_cast<double>(placement_dist_dbu) / length_dbu;
      const int buf_x = static_cast<int>(std::lround(
          to_x + fraction * (from_x - to_x)));
      const int buf_y = static_cast<int>(std::lround(
          to_y + fraction * (from_y - to_y)));
      BufferAction action;
      action.net_id = net_id_;
      action.driver_pin = driver_;
      action.downstream_pins = child.load_pins;
      action.buffer_cell_index = buffer_cell;
      action.x_dbu = buf_x;
      action.y_dbu = buf_y;
      action.downstream_load = buffer_load_cap;
      action.initial_buffer_cell_index = selection.initial_cell_index;
      action.initial_downstream_load = load_cap;
      action.target_load = rebufferTargetLoadForCell(
          liberty_.cells[buffer_cell],
          characterization_.target_slew[0],
          characterization_.target_slew[1],
          config_.repair_design_target_cell_math);
      action.reason = wire_length_violation
                          ? "wire_length"
                          : (cap_violation ? "wire_cap" : "wire_slew");
      actions_->push_back(std::move(action));

      ref_cap = liberty_.cells[buffer_cell].input_cap;
      fanout = 1;
      length_dbu -= buf_dist_dbu;
      wire_length_ref = 0;
      to_x = buf_x;
      to_y = buf_y;
      updateLength(length_dbu, wire_res, wire_cap);
      load_cap = wire_cap + ref_cap;
      // The inserted repeater's input limit becomes the new load-slew budget
      // for the remaining upstream segment.
      max_load_slew = liberty_.cells[buffer_cell].max_slew > 0.0
                          ? liberty_.cells[buffer_cell].max_slew
                          : kInf;
      load_slew = estimateSlew(wire_length_ref, wire_res, wire_cap, ref_cap);
      target_buffer_cell = rebufferChooseBufferCell(
          liberty_,
          characterization_,
          driver_slew_rise_,
          driver_slew_fall_,
          load_cap,
          config_.repair_design_target_cell_math,
          config_.repair_design_target_cell_selection,
          -1,
          config_.sizing_area_limit,
          config_.sizing_leakage_limit);
      if (buf_dist_dbu <= 0 && ref_cap >= prev_ref_cap) {
        if (++zero_progress_iters >= 2) {
          break;
        }
      } else {
        zero_progress_iters = 0;
      }
    }

    wire.wire_length = wire_length_ref + length_dbu;
    wire.cap = load_cap;
    wire.max_load_slew = max_load_slew;
    wire.fanout = fanout;
    wire.load_pins = child.load_pins;
  }

  void repairJunction(int index)
  {
    NetNode& junction = nodes_[index];
    for (const int child_index : junction.children) {
      repairNode(child_index);
    }
    const NetNode& left = nodes_[junction.children[0]];
    const NetNode& right = nodes_[junction.children[1]];
    const double cap = left.cap + right.cap;
    const double max_load_slew
        = std::min(left.max_load_slew, right.max_load_slew);
    const long long combined_wire_length
        = static_cast<long long>(left.wire_length) + right.wire_length;
    const double combined_wire_res
        = (static_cast<double>(combined_wire_length)
           / static_cast<double>(design_.dbu))
          * config_.r_unit;

    // OpenROAD's repairNetJunc combines the two branches with the raw driver
    // resistance plus accumulated downstream wire resistance.
    const double load_slew
        = (driver_resistance_ + combined_wire_res) * cap
          * config_.slew_rc_factor;
    const double max_load_slew_margined
        = max_load_slew * (1.0 - config_.slew_margin);
    const bool load_slew_violation
        = std::isfinite(max_load_slew_margined)
          && load_slew > max_load_slew_margined;
    const bool cap_violation
        = std::isfinite(max_cap_) && max_cap_ > 0.0 && cap > max_cap_;
    const bool wire_length_violation
        = max_wire_length_dbu_ > 0
          && combined_wire_length > max_wire_length_dbu_;

    // OpenROAD evaluates each constraint against the same original branches.
    // Different constraints can isolate both sides; ties select the right.
    std::array<bool, 2> repeat{};
    if (load_slew_violation) {
      const double left_wire_res
          = (static_cast<double>(left.wire_length)
             / static_cast<double>(design_.dbu))
            * config_.r_unit;
      const double right_wire_res
          = (static_cast<double>(right.wire_length)
             / static_cast<double>(design_.dbu))
            * config_.r_unit;
      const double left_slack
          = left.max_load_slew * (1.0 - config_.slew_margin)
            - (driver_resistance_ + left_wire_res) * left.cap
                  * config_.slew_rc_factor;
      const double right_slack
          = right.max_load_slew * (1.0 - config_.slew_margin)
            - (driver_resistance_ + right_wire_res) * right.cap
                  * config_.slew_rc_factor;
      repeat[left_slack < right_slack ? 0 : 1] = true;
    }
    if (cap_violation) {
      repeat[left.cap > right.cap ? 0 : 1] = true;
    }
    if (wire_length_violation) {
      repeat[left.wire_length > right.wire_length ? 0 : 1] = true;
    }
    for (std::size_t branch = 0; branch < repeat.size(); ++branch) {
      if (repeat[branch]) {
        NetNode& child = nodes_[junction.children[branch]];
        const auto selection = rebufferChooseRepeaterCell(
            liberty_,
            characterization_,
            driver_slew_rise_,
            driver_slew_fall_,
            cap,
            child.cap,
            config_);
        const int buffer_cell = selection.cell_index;
        if (buffer_cell >= 0) {
          BufferAction action;
          action.net_id = net_id_;
          action.driver_pin = driver_;
          action.downstream_pins = child.load_pins;
          action.buffer_cell_index = buffer_cell;
          action.x_dbu = junction.x;
          action.y_dbu = junction.y;
          action.downstream_load = child.cap;
          action.initial_buffer_cell_index = selection.initial_cell_index;
          action.initial_downstream_load = cap;
          action.target_load = rebufferTargetLoadForCell(
              liberty_.cells[buffer_cell],
              characterization_.target_slew[0],
              characterization_.target_slew[1],
              config_.repair_design_target_cell_math);
          action.reason = wire_length_violation
                              ? "junction_length"
                              : (cap_violation ? "junction_cap"
                                                : "junction_slew");
          actions_->push_back(std::move(action));
          const auto& buffer = liberty_.cells[buffer_cell];
          child.cap = buffer.input_cap;
          child.fanout = 1;
          child.max_load_slew = buffer.max_slew > 0.0 ? buffer.max_slew : kInf;
          child.wire_length = 0;
        }
      }
    }

    // OpenROAD makeRepeater replaces the isolated branch with its input
    // capacitance, fanout and slew limit before repairNetJunc returns. The
    // upstream wire must see that state rather than the old downstream load.
    junction.cap = left.cap + right.cap;
    junction.fanout = left.fanout + right.fanout;
    junction.max_load_slew
        = std::min(left.max_load_slew, right.max_load_slew);
    junction.wire_length = left.wire_length + right.wire_length;
    junction.load_pins = left.load_pins;
    junction.load_pins.insert(junction.load_pins.end(),
                              right.load_pins.begin(), right.load_pins.end());
  }

  void updateLength(int length_dbu, double& wire_res, double& wire_cap) const
  {
    const double length_um
        = static_cast<double>(length_dbu) / static_cast<double>(design_.dbu);
    wire_res = length_um * config_.r_unit;
    wire_cap = length_um * config_.c_unit;
  }

  double estimateSlew(int wire_length_ref_dbu,
                      double wire_res,
                      double wire_cap,
                      double ref_cap) const
  {
    const double wire_res_ref
        = (static_cast<double>(wire_length_ref_dbu)
           / static_cast<double>(design_.dbu))
          * config_.r_unit;
    return (driver_wire_resistance_ * (wire_cap + ref_cap)
            + wire_res * (wire_cap / 2.0 + ref_cap)
            + wire_res_ref * ref_cap)
           * config_.slew_rc_factor;
  }

  double solveSlewLength(int wire_length_ref_dbu,
                         double ref_cap,
                         double max_load_slew,
                         int buffer_cell) const
  {
    double r_buffer = 0.0;
    if (config_.repair_design_openroad_repeater_resize
        && buffer_cell >= 0
        && buffer_cell < static_cast<int>(liberty_.cells.size())) {
      r_buffer = liberty_.cells[buffer_cell].drive_resistance;
      if (r_buffer <= 0.0) {
        r_buffer = cellDriveResistance(liberty_.cells[buffer_cell]);
      }
    }
    if (r_buffer <= 0.0) {
      r_buffer = driver_wire_resistance_;
    }
    const double wire_length_ref_um
        = config_.repair_design_openroad_repeater_resize
              ? static_cast<double>(wire_length_ref_dbu)
                    / static_cast<double>(design_.dbu)
              : 0.0;
    const double a = config_.repair_design_openroad_repeater_resize
                         ? (config_.r_unit * config_.c_unit)
                         : (config_.r_unit * config_.c_unit / 2.0);
    const double b = r_buffer * config_.c_unit
                     + config_.r_unit * ref_cap
                     + config_.r_unit * config_.c_unit * wire_length_ref_um;
    const double c = r_buffer * ref_cap
                     + wire_length_ref_um * config_.r_unit * ref_cap
                     - max_load_slew / config_.slew_rc_factor;
    if (a > 1e-12) {
      const double discriminant = b * b - 4.0 * a * c;
      if (discriminant < 0.0) {
        return -1.0;
      }
      return (-b + std::sqrt(discriminant)) / (2.0 * a);
    }
    if (b > 1e-12) {
      return -c / b;
    }
    return -1.0;
  }

  const DesignView& design_;
  const TimingView& timing_;
  const SteinerView& steiner_;
  const LibertyView& liberty_;
  const CharacterizationResult& characterization_;
  const RepairConfig& config_;
  int net_id_ = -1;
  int driver_ = -1;
  double max_cap_ = kInf;
  int max_wire_length_dbu_ = 0;
  const std::vector<std::vector<int>>& children_;
  std::vector<NetNode> nodes_;
  std::vector<BufferAction>* actions_ = nullptr;
  double driver_resistance_ = 0.0;
  double driver_wire_resistance_ = 0.0;
  double driver_slew_rise_ = 0.0;
  double driver_slew_fall_ = 0.0;
};

}  // namespace

bool repairNetTree(const DesignView& design,
                   const TimingView& timing,
                   const SteinerView& steiner,
                   const LibertyView& liberty,
                   const CharacterizationResult& characterization,
                   const RepairConfig& config,
                   int net_id,
                   double max_cap,
                   int driver,
                   int driver_cell_index,
                   const std::vector<std::vector<int>>& children,
                   std::vector<BufferAction>& actions,
                   std::string& reason,
                   double* root_cap_out)
{
  if (driver < 0 || driver >= static_cast<int>(children.size())) {
    reason = "invalid_driver_node";
    return false;
  }
  NetTreeRepair repair(design,
                       timing,
                       steiner,
                       liberty,
                       characterization,
                       config,
                       net_id,
                       driver,
                       driver_cell_index,
                       max_cap,
                       children);
  return repair.run(actions, reason, root_cap_out);
}

}  // namespace rebuffer

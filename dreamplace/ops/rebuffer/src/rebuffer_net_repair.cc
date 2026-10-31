// SPDX-License-Identifier: BSD-3-Clause
// OpenROAD-style recursive repairNet pass over one Steiner tree.

#include "rebuffer_repair.hh"

#include <algorithm>
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
  int fanout = 0;
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
    const int lib_pin = rebufferLibPinIndex(design_, driver_);
    if (lib_pin >= 0
        && lib_pin < static_cast<int>(design_.pin_drive_resistance.size())) {
      driver_resistance_ = design_.pin_drive_resistance[lib_pin];
    }
    const int driver_node = design_.pin2node_map[driver_];
    if (driver_node >= 0
        && driver_node < static_cast<int>(design_.inst_main_id.size())) {
      const int main_id = design_.inst_main_id[driver_node];
      if (main_id >= 0
          && main_id + 1
                 < static_cast<int>(design_.main_id_2_cell_id_start.size())) {
        const int cell_index
            = design_.main_id_2_cell_id_start[main_id]
              + design_.inst_libcell_offset[driver_node];
        if (cell_index >= 0
            && cell_index < static_cast<int>(liberty_.cells.size())) {
          const double liberty_resistance
              = cellDriveResistance(liberty_.cells[cell_index]);
          if (liberty_resistance > 0.0) {
            driver_resistance_ = liberty_resistance;
          }
        }
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

  bool run(std::vector<BufferAction>& actions, std::string& reason)
  {
    actions_ = &actions;
    const int root = buildNode(driver_);
    if (root < 0) {
      return true;
    }
    repairNode(root);
    return true;
  }

 private:
  int buildNode(int node)
  {
    if (node < 0 || node >= static_cast<int>(children_.size())) {
      return -1;
    }
    const auto& children = children_[node];
    if (children.empty()) {
      if (node >= steiner_.steiner_node_base || node == driver_) {
        return -1;
      }
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
      load.fanout = 1;
      load.load_pins.push_back(node);
      nodes_.push_back(std::move(load));
      return static_cast<int>(nodes_.size()) - 1;
    }
    if (children.size() == 1) {
      return buildEdge(node, children.front());
    }
    NetNode junction;
    junction.type = NetNode::Type::Junction;
    junction.original_node = node;
    junction.x = static_cast<int>(std::lround(steiner_.node_x[node]));
    junction.y = static_cast<int>(std::lround(steiner_.node_y[node]));
    const int junction_index = static_cast<int>(nodes_.size());
    nodes_.push_back(std::move(junction));
    for (const int child : children) {
      const int edge = buildEdge(node, child);
      if (edge < 0) {
        return -1;
      }
      nodes_[junction_index].children.push_back(edge);
    }
    return junction_index;
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
    double load_slew = estimateSlew(wire_res, wire_cap, ref_cap);
    double max_load_slew = child.max_load_slew;
    double writeback_max_load_slew
        = child.max_load_slew
          - wire_res * (wire_cap / 2.0 + ref_cap) * config_.slew_rc_factor;

    // OpenROAD backs up the segment inserting repeaters until the remaining
    // upstream wire meets max_cap / max_load_slew.  Each repeater becomes the
    // new driver reference for the next iteration.
    int zero_progress_iters = 0;
    while (length_dbu > 0) {
      const double max_load_slew_margined
          = max_load_slew * (1.0 - config_.slew_margin);
      const bool cap_violation
          = std::isfinite(max_cap_) && max_cap_ > 0.0 && load_cap > max_cap_;
      const bool negative_limit = max_load_slew < 0.0;
      const bool slew_violation
          = std::isfinite(max_load_slew_margined)
            && load_slew > max_load_slew_margined;
      if (!negative_limit && !cap_violation && !slew_violation) {
        break;
      }
      int split_length_dbu = length_dbu;
      if (negative_limit) {
        split_length_dbu = 0;
      }
      if (cap_violation && config_.c_unit > 0.0) {
        const double cap_split_um
            = std::max((max_cap_ - ref_cap) / config_.c_unit, 0.0);
        split_length_dbu = std::min(
            split_length_dbu,
            static_cast<int>(std::lround(cap_split_um * design_.dbu)));
      }
      if (slew_violation) {
        const double slew_split_um = solveSlewLength(ref_cap,
                                                     max_load_slew_margined);
        if (slew_split_um >= 0.0) {
          split_length_dbu = std::min(
              split_length_dbu,
              static_cast<int>(std::lround(slew_split_um * design_.dbu)));
        } else {
          split_length_dbu = 0;
        }
      }
      if (split_length_dbu < 0) {
        split_length_dbu = 0;
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
      const int buffer_cell = rebufferChooseBufferCell(
          liberty_,
          characterization_,
          driver_slew_rise_,
          driver_slew_fall_,
          load_cap);
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
      action.target_load = rebufferTargetLoadForCell(
          liberty_.cells[buffer_cell],
          characterization_.target_slew[0],
          characterization_.target_slew[1]);
      action.reason = cap_violation ? "wire_cap" : "wire_slew";
      actions_->push_back(std::move(action));

      ref_cap = liberty_.cells[buffer_cell].input_cap;
      length_dbu -= buf_dist_dbu;
      to_x = buf_x;
      to_y = buf_y;
      updateLength(length_dbu, wire_res, wire_cap);
      load_cap = wire_cap + ref_cap;
      load_slew = estimateSlew(wire_res, wire_cap, ref_cap);
      // The inserted repeater's input limit becomes the new load-slew budget
      // for the remaining upstream segment.
      if (liberty_.cells[buffer_cell].max_slew > 0.0) {
        max_load_slew = liberty_.cells[buffer_cell].max_slew;
      }
      writeback_max_load_slew
          = max_load_slew
            - wire_res * (wire_cap / 2.0 + ref_cap) * config_.slew_rc_factor;
      if (buf_dist_dbu <= 0 && ref_cap >= prev_ref_cap) {
        if (++zero_progress_iters >= 2) {
          break;
        }
      } else {
        zero_progress_iters = 0;
      }
    }

    wire.cap = load_cap;
    wire.max_load_slew = writeback_max_load_slew;
    wire.fanout = child.fanout;
    wire.load_pins = child.load_pins;
  }

  void repairJunction(int index)
  {
    NetNode& junction = nodes_[index];
    double cap = 0.0;
    double max_load_slew = kInf;
    int fanout = 0;
    std::vector<int> load_pins;
    for (const int child_index : junction.children) {
      repairNode(child_index);
      const NetNode& child = nodes_[child_index];
      cap += child.cap;
      max_load_slew = std::min(max_load_slew, child.max_load_slew);
      fanout += child.fanout;
      load_pins.insert(
          load_pins.end(), child.load_pins.begin(), child.load_pins.end());
    }

    // OpenROAD's repairNetJunc combines the two branches with the raw driver
    // resistance.  Driver slew is handled before this pass by resize /
    // findSlewLoadCap; it is not re-checked at every junction.
    const double load_slew = driver_resistance_ * cap * config_.slew_rc_factor;
    const double max_load_slew_margined
        = max_load_slew * (1.0 - config_.slew_margin);
    const bool load_slew_violation
        = std::isfinite(max_load_slew_margined)
          && load_slew > max_load_slew_margined;
    const bool cap_violation
        = std::isfinite(max_cap_) && max_cap_ > 0.0 && cap > max_cap_;

    if (load_slew_violation || cap_violation) {
      int selected = -1;
      if (cap_violation) {
        double selected_cap = -1.0;
        for (const int child_index : junction.children) {
          const NetNode& child = nodes_[child_index];
          if (child.cap > selected_cap) {
            selected_cap = child.cap;
            selected = child_index;
          }
        }
      } else {
        double selected_slack = kInf;
        for (const int child_index : junction.children) {
          const NetNode& child = nodes_[child_index];
          const double child_limit
              = child.max_load_slew * (1.0 - config_.slew_margin);
          const double child_slew
              = driver_resistance_ * child.cap * config_.slew_rc_factor;
          const double slack = child_limit - child_slew;
          if (slack < selected_slack) {
            selected_slack = slack;
            selected = child_index;
          }
        }
      }
      if (selected >= 0) {
        const NetNode& child = nodes_[selected];
        const int buffer_cell = rebufferChooseBufferCell(
            liberty_,
            characterization_,
            driver_slew_rise_,
            driver_slew_fall_,
            cap);
        if (buffer_cell >= 0) {
          BufferAction action;
          action.net_id = net_id_;
          action.driver_pin = driver_;
          action.downstream_pins = child.load_pins;
          action.buffer_cell_index = buffer_cell;
          action.x_dbu = junction.x;
          action.y_dbu = junction.y;
          action.downstream_load = child.cap;
          action.target_load = rebufferTargetLoadForCell(
              liberty_.cells[buffer_cell],
              characterization_.target_slew[0],
              characterization_.target_slew[1]);
          action.reason = cap_violation ? "junction_cap" : "junction_slew";
          actions_->push_back(std::move(action));
        }
      }
    }

    // Keep the pre-insertion branch capacitance and limit, matching
    // OpenROAD's BufferedNet write-back.  The inserted buffer's input cap is
    // accounted for only when the parent wire backs up past it.
    junction.cap = cap;
    junction.max_load_slew = max_load_slew;
    junction.fanout = fanout;
    junction.load_pins = std::move(load_pins);
  }

  void updateLength(int length_dbu, double& wire_res, double& wire_cap) const
  {
    const double length_um
        = static_cast<double>(length_dbu) / static_cast<double>(design_.dbu);
    wire_res = length_um * config_.r_unit;
    wire_cap = length_um * config_.c_unit;
  }

  double estimateSlew(double wire_res,
                      double wire_cap,
                      double ref_cap) const
  {
    return (driver_wire_resistance_ * (wire_cap + ref_cap)
            + wire_res * ref_cap + wire_res * wire_cap / 2.0)
           * config_.slew_rc_factor;
  }

  double solveSlewLength(double ref_cap, double max_load_slew) const
  {
    const double a = config_.r_unit * config_.c_unit / 2.0;
    const double b = driver_wire_resistance_ * config_.c_unit
                     + config_.r_unit * ref_cap;
    const double c = driver_wire_resistance_ * ref_cap
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
                   const std::vector<std::vector<int>>& children,
                   std::vector<BufferAction>& actions,
                   std::string& reason)
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
                       max_cap,
                       children);
  return repair.run(actions, reason);
}

}  // namespace rebuffer

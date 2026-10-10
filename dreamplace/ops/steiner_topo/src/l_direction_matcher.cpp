// Batched route-wire matching. Routing feedback is prepared once; tree geometry
// is live.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace py = pybind11;
namespace {
using Point = std::array<double, 2>;
using Key = std::pair<int64_t, int64_t>;
using AxisIndices = std::array<int, 4>;
using Clock = std::chrono::steady_clock;
constexpr int Unknown = -1, HFirst = 0, VFirst = 1, Straight = 2,
              FakeStraight = 3;

Key key(Point p) {
  // Python round uses ties-to-even for the existing microunit endpoint index.
  return {static_cast<int64_t>(std::nearbyint(p[0] * 1e6)),
          static_cast<int64_t>(std::nearbyint(p[1] * 1e6))};
}
struct Wire {
  Point a, b;
  bool horizontal, vertical;
  int order;
};
struct Net {
  bool present = false, snapped = false;
  std::vector<Wire> wires;
  std::map<Key, std::vector<int>> endpoints;
  std::optional<Point> shared;
};

int axisIndex(const std::vector<double> &axis, double value, bool upper_tie) {
  if (axis.empty())
    return -1;
  auto found = std::lower_bound(axis.begin(), axis.end(), value);
  int upper = static_cast<int>(found - axis.begin());
  if (upper == 0)
    return 0;
  if (upper == static_cast<int>(axis.size()))
    return upper - 1;
  double high = std::abs(axis[upper] - value),
         low = std::abs(axis[upper - 1] - value);
  return (high < low || (upper_tie && high == low)) ? upper : upper - 1;
}

class LDirectionMatcher {
public:
  LDirectionMatcher(const std::vector<std::string> &names, py::dict routes,
                    std::vector<double> x_axis, std::vector<double> y_axis,
                    std::vector<double> tolerances, double endpoint_limit)
      : nets_(names.size()), x_(std::move(x_axis)), y_(std::move(y_axis)),
        tolerances_(std::move(tolerances)), endpoint_limit_(endpoint_limit) {
    for (size_t n = 0; n < names.size(); ++n) {
      py::str name(names[n]);
      if (!routes.contains(name))
        continue;
      auto data = routes[name].cast<py::dict>();
      auto &net = nets_[n];
      net.present = true;
      net.snapped = data.contains("source") &&
                    data["source"].cast<std::string>() == "gpugr";
      std::vector<std::pair<Key, Point>> points;
      std::map<Key, int> counts;
      if (data.contains("wires")) {
        for (auto item : data["wires"].cast<py::list>()) {
          auto w = item.cast<py::dict>();
          Wire wire{w["real1"].cast<Point>(), w["real2"].cast<Point>(),
                    w.contains("is_horizontal") &&
                        w["is_horizontal"].cast<bool>(),
                    w.contains("is_vertical") && w["is_vertical"].cast<bool>(),
                    w.contains("order") ? w["order"].cast<int>() : 0};
          int index = static_cast<int>(net.wires.size());
          net.wires.push_back(wire);
          for (Point p : {wire.a, wire.b}) {
            auto k = key(p);
            net.endpoints[k].push_back(index);
            if (++counts[k] == 1)
              points.push_back({k, p});
          }
        }
      }
      for (const auto &p : points) {
        if (counts[p.first] > 1) {
          net.shared = p.second;
          break;
        }
      }
    }
  }

  py::tuple
  resolve(py::array_t<int64_t, py::array::c_style | py::array::forcecast> from,
          py::array_t<int64_t, py::array::c_style | py::array::forcecast> to,
          py::array_t<int64_t, py::array::c_style | py::array::forcecast>
              vertex_net,
          py::array_t<double, py::array::c_style | py::array::forcecast> x,
          py::array_t<double, py::array::c_style | py::array::forcecast> y,
          bool single_precision) const {
    if (from.ndim() != 1 || to.ndim() != 1 || x.ndim() != 1 || y.ndim() != 1 ||
        vertex_net.ndim() != 1 || from.size() != to.size() ||
        x.size() != y.size() || x.size() != vertex_net.size())
      throw py::value_error("L-direction array domains mismatch");
    // Explicit strides also support the bundled pybind11 with NumPy 2.
    py::array_t<int32_t> output({from.size()},
                                {static_cast<py::ssize_t>(sizeof(int32_t))});
    auto a = from.unchecked<1>(), b = to.unchecked<1>(),
         net_id = vertex_net.unchecked<1>();
    auto px = x.unchecked<1>(), pyv = y.unchecked<1>();
    auto result = output.mutable_unchecked<1>();
    std::vector<std::pair<int64_t, AxisIndices>> records;
    double edge_ms, fallback_ms;
    int resolved = 0, downgraded = 0;
    {
      py::gil_scoped_release release;
      auto start = Clock::now();
      records.reserve(from.size());
      for (int64_t edge = 0; edge < from.size(); ++edge) {
        result(edge) = Unknown;
        if (a(edge) < 0 || b(edge) < 0)
          continue;
        if (a(edge) >= x.size() || b(edge) >= x.size())
          throw std::invalid_argument(
              "L-direction edge vertex outside geometry");
        auto n = net_id(a(edge));
        if (n < 0 || n >= static_cast<int64_t>(nets_.size()) ||
            !nets_[n].present)
          continue;
        const auto &net = nets_[n];
        Point p{px(a(edge)), pyv(a(edge))}, q{px(b(edge)), pyv(b(edge))};
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) ||
            !std::isfinite(q[0]) || !std::isfinite(q[1]))
          throw std::invalid_argument("L-direction geometry must be finite");
        result(edge) = direction(net, p, q, single_precision);
        p = snap(net, p);
        q = snap(net, q);
        records.push_back(
            {edge,
             {axisIndex(x_, p[0], false), axisIndex(y_, p[1], false),
              axisIndex(x_, q[0], false), axisIndex(y_, q[1], false)}});
      }
      auto matched = Clock::now();
      edge_ms =
          std::chrono::duration<double, std::milli>(matched - start).count();
      if (!x_.empty() && !y_.empty()) {
        // Known paths contribute once. Resolving an unknown never changes this
        // map.
        std::vector<float> h(x_.size() * y_.size(), 0), v(h.size(), 0);
        for (const auto &record : records) {
          int d = result(record.first);
          const auto &i = record.second;
          if (std::find(i.begin(), i.end(), -1) != i.end())
            continue;
          if (d == HFirst || (d == Straight && i[1] == i[3]))
            addH(h, i[0], i[2], i[1]);
          if (d == HFirst)
            addV(v, i[2], i[1], i[3]);
          if (d == VFirst || (d == Straight && i[1] != i[3] && i[0] == i[2]))
            addV(v, i[0], i[1], i[3]);
          if (d == VFirst)
            addH(h, i[0], i[2], i[3]);
        }
        for (const auto &record : records) {
          auto &d = result(record.first);
          if (d != Unknown && d != FakeStraight)
            continue;
          const auto &i = record.second;
          double cost_h = sumH(h, i[0], i[2], i[1]) + sumV(v, i[2], i[1], i[3]);
          double cost_v = sumV(v, i[0], i[1], i[3]) + sumH(h, i[0], i[2], i[3]);
          if (std::abs(cost_h - cost_v) <= 1e-6) {
            if (d == FakeStraight) {
              d = Unknown;
              ++downgraded;
            }
          } else {
            d = cost_h < cost_v ? HFirst : VFirst;
            ++resolved;
          }
        }
      }
      fallback_ms =
          std::chrono::duration<double, std::milli>(Clock::now() - matched)
              .count();
    }
    return py::make_tuple(output, edge_ms, fallback_ms, resolved, downgraded);
  }

private:
  Point snap(const Net &net, Point p) const {
    if (net.snapped) {
      if (!x_.empty())
        p[0] = x_[axisIndex(x_, p[0], true)];
      if (!y_.empty())
        p[1] = y_[axisIndex(y_, p[1], true)];
    }
    return p;
  }

  int first(const Net &net, Point p, Point target, double tolerance) const {
    double endpoint =
        net.snapped ? std::min(tolerance, endpoint_limit_) : tolerance;
    auto indexed = net.endpoints.find(key(p));
    for (bool endpoint_only : {true, false}) {
      int best = -1;
      using Rank = std::tuple<int, double, double, double, int>;
      Rank best_rank;
      auto consider = [&](int index) {
        const auto &w = net.wires[index];
        if (!w.horizontal && !w.vertical)
          return;
        bool at = (std::abs(w.a[0] - p[0]) <= endpoint &&
                   std::abs(w.a[1] - p[1]) <= endpoint) ||
                  (std::abs(w.b[0] - p[0]) <= endpoint &&
                   std::abs(w.b[1] - p[1]) <= endpoint);
        double lo_x = std::min(w.a[0], w.b[0]), hi_x = std::max(w.a[0], w.b[0]);
        double lo_y = std::min(w.a[1], w.b[1]), hi_y = std::max(w.a[1], w.b[1]);
        bool on = w.horizontal
                      ? std::abs(w.a[1] - p[1]) <= tolerance &&
                            p[0] >= lo_x - tolerance && p[0] <= hi_x + tolerance
                      : std::abs(w.a[0] - p[0]) <= tolerance &&
                            p[1] >= lo_y - tolerance &&
                            p[1] <= hi_y + tolerance;
        if (!at && (endpoint_only || !on))
          return;
        Point close = w.horizontal
                          ? Point{std::clamp(p[0], lo_x, hi_x), w.a[1]}
                          : Point{w.a[0], std::clamp(p[1], lo_y, hi_y)};
        Point next = w.horizontal
                         ? Point{std::clamp(target[0], lo_x, hi_x), w.a[1]}
                         : Point{w.a[0], std::clamp(target[1], lo_y, hi_y)};
        double progress =
            std::abs(target[0] - p[0]) + std::abs(target[1] - p[1]) -
            (std::abs(target[0] - next[0]) + std::abs(target[1] - next[1]));
        Rank rank{at ? 0 : 1, -progress,
                  std::hypot(close[0] - p[0], close[1] - p[1]),
                  std::min(std::hypot(w.a[0] - p[0], w.a[1] - p[1]),
                           std::hypot(w.b[0] - p[0], w.b[1] - p[1])),
                  w.order};
        if (best < 0 || rank < best_rank) {
          best = index;
          best_rank = rank;
        }
      };
      if (endpoint_only && indexed != net.endpoints.end()) {
        for (int i : indexed->second)
          consider(i);
      } else {
        for (size_t i = 0; i < net.wires.size(); ++i)
          consider(static_cast<int>(i));
      }
      if (best >= 0)
        return net.wires[best].horizontal ? HFirst : VFirst;
    }
    return Unknown;
  }

  int direction(const Net &net, Point p, Point q, bool single_precision) const {
    double dx = std::abs(p[0] - q[0]), dy = std::abs(p[1] - q[1]);
    // Preserve NumPy's geometry-dtype arithmetic at epsilon/corner ties.
    double eps = single_precision ? static_cast<double>(1e-5f) : 1e-5;
    if (single_precision) {
      dx = static_cast<float>(dx);
      dy = static_cast<float>(dy);
    }
    if (dx < eps && dy < eps)
      return Unknown;
    if ((dx < eps && dy > eps) || (dy < eps && dx > eps))
      return Straight;
    if (net.wires.size() == 1)
      return FakeStraight;
    auto a = snap(net, p), b = snap(net, q);
    static const std::vector<double> egr_tolerance{1.0};
    for (double t : net.snapped ? tolerances_ : egr_tolerance) {
      int d = first(net, a, b, t);
      if (d != Unknown)
        return d;
      d = first(net, b, a, t);
      if (d != Unknown)
        return d == HFirst ? VFirst : HFirst;
    }
    if (net.shared) {
      auto c = *net.shared;
      double h = std::abs(c[0] - q[0]) + std::abs(c[1] - p[1]);
      double v = std::abs(c[0] - p[0]) + std::abs(c[1] - q[1]);
      if (single_precision) {
        h = std::abs(static_cast<float>(c[0]) - static_cast<float>(q[0])) +
            std::abs(static_cast<float>(c[1]) - static_cast<float>(p[1]));
        v = std::abs(static_cast<float>(c[0]) - static_cast<float>(p[0])) +
            std::abs(static_cast<float>(c[1]) - static_cast<float>(q[1]));
      }
      return h < v ? HFirst : VFirst;
    }
    return Unknown;
  }

  void addH(std::vector<float> &map, int a, int b, int y) const {
    for (int x = std::min(a, b); x <= std::max(a, b); ++x)
      map[x * y_.size() + y] += 1;
  }
  void addV(std::vector<float> &map, int x, int a, int b) const {
    for (int y = std::min(a, b); y <= std::max(a, b); ++y)
      map[x * y_.size() + y] += 1;
  }
  double sumH(const std::vector<float> &map, int a, int b, int y) const {
    double sum = 0;
    for (int x = std::min(a, b); x <= std::max(a, b); ++x)
      sum += map[x * y_.size() + y];
    return sum;
  }
  double sumV(const std::vector<float> &map, int x, int a, int b) const {
    double sum = 0;
    for (int y = std::min(a, b); y <= std::max(a, b); ++y)
      sum += map[x * y_.size() + y];
    return sum;
  }
  std::vector<Net> nets_;
  std::vector<double> x_, y_, tolerances_;
  double endpoint_limit_;
};
} // namespace

void bindLDirectionMatcher(py::module_ &m) {
  py::class_<LDirectionMatcher>(m, "LDirectionMatcher")
      .def(py::init<const std::vector<std::string> &, py::dict,
                    std::vector<double>, std::vector<double>,
                    std::vector<double>, double>())
      .def("resolve", &LDirectionMatcher::resolve);
}

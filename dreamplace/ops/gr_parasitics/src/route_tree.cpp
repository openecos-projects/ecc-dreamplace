#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace py = pybind11;
namespace {
using IndexArray = py::array_t<int32_t, py::array::c_style>;
using ValueArray = py::array_t<double, py::array::c_style>;

template <class T>
std::vector<T> vector(const py::array_t<T, py::array::c_style>& input) {
    if (input.ndim() != 1 || input.size() > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("GR pack requires flat arrays within int32 capacity");
    }
    return {input.data(), input.data() + input.size()};
}

std::vector<int32_t> indices(const py::dict& pack, const char* name) {
    auto input = IndexArray::ensure(pack[name]);
    if (!input) throw std::runtime_error(std::string("GR pack field must be contiguous int32: ") + name);
    return vector(input);
}

template <class T>
py::array_t<T> array(const std::vector<T>& input) {
    py::array_t<T> result(input.size());
    std::copy(input.begin(), input.end(), result.mutable_data());
    return result;
}

int mapped(const py::dict& names, const std::string& name) {
    const py::str key(name);
    return names.contains(key) ? names[key].cast<int>() : -1;
}

void checkOffsets(const std::vector<int32_t>& offsets, int count, int end) {
    if (offsets.size() != static_cast<size_t>(count + 1) || offsets.front() != 0 ||
        offsets.back() != end || !std::is_sorted(offsets.begin(), offsets.end())) {
        throw std::runtime_error("Malformed GR net CSR offsets");
    }
}

struct Edge {
    int from, to, raw;
    double resistance, capacitance;
};

py::dict prepareTree(const py::dict& route, const py::dict& pinNames, const py::dict& netNames,
                     IndexArray pyPinNetArray, IndexArray driverArray, IndexArray eligibleArray,
                     ValueArray rArray, ValueArray cArray, ValueArray viaArray) {
    const auto pyPinNet = vector(pyPinNetArray), drivers = vector(driverArray), eligible = vector(eligibleArray);
    const auto r = vector(rArray), c = vector(cArray), via = vector(viaArray);
    const auto names = route["net_names"].cast<std::vector<std::string>>();
    const auto pins = route["pin_names"].cast<std::vector<std::string>>();
    const auto rawNet = indices(route, "vertex_net"), layer = indices(route, "vertex_layer");
    const auto x = indices(route, "vertex_x_dbu"), y = indices(route, "vertex_y_dbu");
    const auto from = indices(route, "edge_from"), to = indices(route, "edge_to"), kind = indices(route, "edge_kind");
    const auto vstart = indices(route, "net_vertex_start"), estart = indices(route, "net_edge_start");
    const auto status = indices(route, "net_status"), pinVertex = indices(route, "pin_vertex");
    const auto pinNet = indices(route, "pin_net"), pinDriver = indices(route, "pin_is_driver");
    const auto pinLayer = indices(route, "pin_access_layer"), physicalLayer = indices(route, "pin_physical_layer");
    const auto pinX = indices(route, "pin_x_dbu"), pinY = indices(route, "pin_y_dbu");
    const double dbu = route["dbu_per_micron"].cast<double>();
    const int V = rawNet.size(), E = from.size(), N = names.size(), P = pyPinNet.size();
    if (!std::isfinite(dbu) || dbu <= 0 || drivers.size() != eligible.size() ||
        r.size() != c.size() || via.size() + 1 != r.size() || layer.size() != rawNet.size() ||
        x.size() != rawNet.size() || y.size() != rawNet.size() || from.size() != to.size() ||
        kind.size() != from.size() || status.size() != names.size() ||
        pinNet.size() != pins.size() || pinVertex.size() != pins.size() ||
        pinDriver.size() != pins.size() || pinLayer.size() != pins.size() ||
        physicalLayer.size() != pins.size() || pinX.size() != pins.size() || pinY.size() != pins.size()) {
        throw std::runtime_error("GR pack domain/shape/unit mismatch");
    }
    checkOffsets(vstart, N, V);
    checkOffsets(estart, N, E);
    std::vector<std::vector<int>> netPins(N), adjacency;
    for (int p = 0; p < static_cast<int>(pins.size()); ++p) {
        if (pinNet[p] < 0 || pinNet[p] >= N) throw std::runtime_error("GR pin net out of range");
        netPins[pinNet[p]].push_back(p);
    }
    for (int e = 0; e < E; ++e) {
        if (from[e] < 0 || from[e] >= V || to[e] < 0 || to[e] >= V ||
            from[e] == to[e] || rawNet[from[e]] != rawNet[to[e]]) {
            throw std::runtime_error("Invalid GR edge connectivity");
        }
    }
    std::vector<int32_t> pinMap(P, -1), gridMap(P, -1), oldToNew(V, -1);
    std::vector<int32_t> vertexNet, vertexLayer, parent, topo, topoStart{0}, roots, netIds;
    std::vector<int32_t> edgeParent, edgeChild;
    std::vector<double> xy, incomingR, wireCap, edgeR, edgeC;
    std::vector<int> covered(eligible.size(), 0), mappedPins(P, 0);
    std::vector<Edge> graph;
    std::vector<bool> used;
    py::list loopEdges;
    int attachmentCount = 0, attachmentVias = 0;
    double attachmentLength = 0, attachmentResistance = 0, attachmentCap = 0;
    auto vertex = [&](int net, int metal, double px, double py) {
        if (parent.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("GR electrical vertices exceed int32 capacity");
        }
        int id = parent.size();
        vertexNet.push_back(net); vertexLayer.push_back(metal);
        xy.push_back(px); xy.push_back(py);
        parent.push_back(-2); incomingR.push_back(0); wireCap.push_back(0);
        adjacency.emplace_back();
        return id;
    };
    auto edge = [&](int a, int b, double resistance, double capacitance, int raw) {
        if (graph.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("GR electrical edges exceed int32 capacity");
        }
        int id = graph.size();
        graph.push_back({a, b, raw, resistance, capacitance}); used.push_back(false);
        adjacency[a].push_back(id); adjacency[b].push_back(id);
        wireCap[a] += capacitance / 2; wireCap[b] += capacitance / 2;
    };
    int filtered = 0;
    for (int n = 0; n < N; ++n) {
        const int target = mapped(netNames, names[n]);
        if (target < 0 || target >= static_cast<int>(eligible.size()) || !eligible[target]) {
            ++filtered;
            continue;
        }
        auto fail = [&](const std::string& reason) { throw std::runtime_error("GR net '" + names[n] + "': " + reason); };
        if (covered[target]++) fail("duplicate PyDB net mapping");
        if (status[n] >= 2) fail("failed or unrouted");
        if (vstart[n] == vstart[n + 1]) fail("missing route vertices");
        const size_t netBegin = parent.size();
        for (int old = vstart[n]; old < vstart[n + 1]; ++old) {
            if (rawNet[old] != n || layer[old] < 0 || layer[old] >= static_cast<int>(r.size())) {
                fail("vertex net/layer mismatch");
            }
            oldToNew[old] = vertex(target, layer[old], x[old] / dbu, y[old] / dbu);
        }
        int root = -1, driverCount = 0;
        for (int p : netPins[n]) {
            const int targetPin = mapped(pinNames, pins[p]);
            if (targetPin < 0 || targetPin >= P || pyPinNet[targetPin] != target || mappedPins[targetPin]++) {
                fail("missing/duplicate/mismatched PyDB pin mapping");
            }
            if (pinVertex[p] < vstart[n] || pinVertex[p] >= vstart[n + 1]) fail("missing pin attachment");
            const int metal = pinLayer[p], physical = physicalLayer[p], grid = pinVertex[p];
            if (metal != layer[grid] || physical < 0 || physical >= static_cast<int>(r.size())) {
                fail("invalid pin access layer");
            }
            if (!std::isfinite(r[metal]) || r[metal] <= 0 || !std::isfinite(c[metal]) || c[metal] < 0) {
                fail("missing or invalid attachment layer RC");
            }
            const double length = (std::abs(static_cast<double>(pinX[p]) - x[grid]) +
                                   std::abs(static_cast<double>(pinY[p]) - y[grid])) / dbu;
            double resistance = length * r[metal];
            for (int l = std::min(metal, physical); l < std::max(metal, physical); ++l) {
                if (!std::isfinite(via[l]) || via[l] <= 0) fail("missing/ambiguous access via resistance");
                resistance += via[l]; ++attachmentVias;
            }
            const double capacitance = length * c[metal];
            gridMap[targetPin] = oldToNew[grid];
            pinMap[targetPin] = vertex(target, physical, pinX[p] / dbu, pinY[p] / dbu);
            edge(pinMap[targetPin], gridMap[targetPin], resistance, capacitance, -1);
            ++attachmentCount; attachmentLength += length;
            attachmentResistance += resistance; attachmentCap += capacitance;
            if (pinDriver[p]) {
                ++driverCount;
                // PyDB derives drivers from timing arcs. A dangling output has
                // no timing arc, but its physical RC tree still has a driver.
                if (drivers[target] >= 0 && drivers[target] != targetPin) fail("native/PyDB driver mismatch");
                root = pinMap[targetPin];
            }
        }
        if (driverCount != 1) fail("requires exactly one actual driver");
        for (int e = estart[n]; e < estart[n + 1]; ++e) {
            const int a = from[e], b = to[e];
            if (a < vstart[n] || a >= vstart[n + 1] || b < vstart[n] || b >= vstart[n + 1]) {
                fail("edge outside net CSR");
            }
            double resistance, capacitance;
            if (kind[e] == 0) {
                if (layer[a] != layer[b] || (x[a] != x[b] && y[a] != y[b])) fail("invalid wire geometry");
                const int l = layer[a];
                if (!std::isfinite(r[l]) || r[l] <= 0 || !std::isfinite(c[l]) || c[l] < 0) {
                    fail("missing or invalid layer wire RC");
                }
                const double length = (std::abs(static_cast<double>(x[a]) - x[b]) +
                                       std::abs(static_cast<double>(y[a]) - y[b])) / dbu;
                if (length <= 0) fail("nonpositive wire length");
                resistance = length * r[l]; capacitance = length * c[l];
            } else if (kind[e] == 1) {
                if (std::abs(layer[a] - layer[b]) != 1 || x[a] != x[b] || y[a] != y[b]) fail("invalid via geometry");
                resistance = via[std::min(layer[a], layer[b])]; capacitance = 0;
                if (!std::isfinite(resistance) || resistance <= 0) fail("missing/ambiguous via resistance");
            } else {
                fail("unsupported edge kind");
            }
            edge(oldToNew[a], oldToNew[b], resistance, capacitance, e);
        }
        roots.push_back(root); netIds.push_back(target);
        parent[root] = -1;
        // Iterative DFS preserves original edge order and avoids recursive
        // stack limits. All graph C was accumulated before omitting loop R.
        std::vector<std::pair<int, size_t>> stack{{root, 0}};
        topo.push_back(root);
        size_t visited = 1;
        while (!stack.empty()) {
            auto& [u, cursor] = stack.back();
            if (cursor == adjacency[u].size()) { stack.pop_back(); continue; }
            int id = adjacency[u][cursor++];
            if (used[id]) continue;
            used[id] = true;
            const auto& connection = graph[id];
            const int v = connection.from == u ? connection.to : connection.from;
            if (parent[v] != -2) {
                int e = connection.raw;
                if (e < 0) fail("unexpected loop through pin attachment");
                py::dict evidence;
                evidence["net_id"] = target; evidence["raw_edge_id"] = e;
                evidence["resistance_ohm"] = connection.resistance;
                evidence["length_um"] = (std::abs(static_cast<double>(x[from[e]]) - x[to[e]]) +
                                          std::abs(static_cast<double>(y[from[e]]) - y[to[e]])) / dbu;
                evidence["from_layer"] = layer[from[e]]; evidence["to_layer"] = layer[to[e]];
                loopEdges.append(evidence);
                continue;
            }
            parent[v] = u; incomingR[v] = connection.resistance;
            edgeParent.push_back(u); edgeChild.push_back(v);
            edgeR.push_back(connection.resistance); edgeC.push_back(connection.capacitance);
            topo.push_back(v); ++visited;
            stack.emplace_back(v, 0);
        }
        if (visited != parent.size() - netBegin) fail("disconnected routing");
        topoStart.push_back(topo.size());
    }
    for (int n = 0; n < static_cast<int>(eligible.size()); ++n) {
        if (eligible[n] && !covered[n]) throw std::runtime_error("Missing eligible PyDB net in GR route");
    }
    for (int p = 0; p < P; ++p) {
        if (pyPinNet[p] < 0 || pyPinNet[p] >= static_cast<int>(eligible.size())) {
            throw std::runtime_error("PyDB pin net out of range");
        }
        if (eligible[pyPinNet[p]] && !mappedPins[p]) throw std::runtime_error("Missing eligible PyDB pin in GR route");
    }
    std::vector<int32_t> childStart(parent.size() + 1, 0), children(edgeChild.size());
    for (int p : edgeParent) ++childStart[p + 1];
    for (size_t i = 1; i < childStart.size(); ++i) childStart[i] += childStart[i - 1];
    auto cursor = childStart;
    for (size_t i = 0; i < edgeChild.size(); ++i) children[cursor[edgeParent[i]]++] = edgeChild[i];
    py::dict result;
    result["pin_to_vertex"] = array(pinMap);
    result["pin_grid_vertex"] = array(gridMap);
    result["vertex_net"] = array(vertexNet); result["vertex_layer"] = array(vertexLayer);
    auto coordinates = py::array_t<double>({static_cast<py::ssize_t>(parent.size()), py::ssize_t(2)});
    std::copy(xy.begin(), xy.end(), coordinates.mutable_data()); result["vertex_xy_um"] = coordinates;
    result["parent"] = array(parent); result["incoming_resistance"] = array(incomingR);
    result["wire_cap"] = array(wireCap); result["child_start"] = array(childStart);
    result["child_vertex"] = array(children); result["topo_order"] = array(topo);
    result["net_topo_start"] = array(topoStart); result["root_vertex"] = array(roots);
    result["net_ids"] = array(netIds); result["edge_parent"] = array(edgeParent);
    result["edge_child"] = array(edgeChild); result["edge_resistance"] = array(edgeR);
    result["edge_capacitance"] = array(edgeC); result["filtered_net_count"] = filtered;
    py::dict model;
    model["pin_access_model"] = "router_selected_gcell_with_estimated_local_rc";
    model["pin_reference"] = "selected_physical_shape_center";
    model["estimated_attachment_count"] = attachmentCount;
    model["attachment_length_um"] = attachmentLength;
    model["attachment_resistance_ohm"] = attachmentResistance;
    model["attachment_capacitance_pf"] = attachmentCap;
    model["attachment_via_count"] = attachmentVias;
    model["electrical_reduction_model"] = "driver_rooted_dfs_tree_with_full_wire_cap";
    model["loop_count"] = loopEdges.size(); model["loop_edges"] = loopEdges;
    result["model_report"] = model;
    return result;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("prepare_tree", &prepareTree, "Prepare GR RC trees with estimated pin access and recorded loop reduction");
}

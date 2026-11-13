#include "spef_tree.h"

#include <pybind11/stl.h>
#include <zlib.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace gr_parasitics {
namespace {

constexpr double kMinConductanceOhm = 1e-9;
constexpr size_t kMaxDenseNameId = 16'000'000;
constexpr size_t kMaxLoopSample = 64;

template <class T>
std::vector<T> toVector(const py::array_t<T, py::array::c_style>& input) {
    if (input.ndim() != 1 || input.size() > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("SPEF pack requires flat arrays within int32 capacity");
    }
    return {input.data(), input.data() + input.size()};
}

template <class T>
py::array_t<T> toArray(const std::vector<T>& input) {
    py::array_t<T> result(input.size());
    std::copy(input.begin(), input.end(), result.mutable_data());
    return result;
}

std::string toUpper(std::string_view input) {
    std::string out(input);
    for (char& ch : out) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return out;
}

std::string unescapeSpefIdent(std::string_view raw) {
    std::string out;
    out.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == '\\' && i + 1 < raw.size()) {
            ++i;
        }
        out.push_back(raw[i]);
    }
    return out;
}

class NameLookup {
public:
    NameLookup(const py::dict& dict, const char* domainName) : domain_(domainName) {
        exact_.reserve(dict.size());
        unescaped_.reserve(dict.size());
        for (const auto& item : dict) {
            const std::string key = py::cast<std::string>(item.first);
            const int32_t id = py::cast<int32_t>(item.second);
            exact_.emplace(key, id);
            const std::string canonical = unescapeSpefIdent(key);
            auto [it, inserted] = unescaped_.emplace(canonical, id);
            if (!inserted && it->second != id) {
                it->second = -2;  // Mark ambiguous unescaped collision.
            }
        }
    }

    int32_t find(std::string_view query) const {
        const std::string key(query);
        if (auto it = exact_.find(key); it != exact_.end()) {
            return it->second;
        }
        const std::string canonical = unescapeSpefIdent(query);
        if (auto it = unescaped_.find(canonical); it != unescaped_.end()) {
            if (it->second == -2) {
                throw std::runtime_error(std::string("Ambiguous escaped ") + domain_ +
                                         " identifier in SPEF: " + key);
            }
            return it->second;
        }
        return -1;
    }

private:
    const char* domain_;
    std::unordered_map<std::string, int32_t> exact_;
    std::unordered_map<std::string, int32_t> unescaped_;
};

class SpefNameMap {
public:
    void insert(uint64_t id, std::string value) {
        if (id < kMaxDenseNameId) {
            if (id >= dense_.size()) {
                dense_.resize(static_cast<size_t>(id) + 1);
            }
            dense_[static_cast<size_t>(id)] = std::move(value);
        } else {
            sparse_[id] = std::move(value);
        }
    }

    const std::string* lookup(uint64_t id) const {
        if (id < dense_.size()) {
            const std::string& entry = dense_[static_cast<size_t>(id)];
            return entry.empty() ? nullptr : &entry;
        }
        auto it = sparse_.find(id);
        return it == sparse_.end() ? nullptr : &it->second;
    }

    std::string expandNetOrPort(std::string_view token) const {
        if (token.size() > 1 && token[0] == '*' &&
            std::isdigit(static_cast<unsigned char>(token[1]))) {
            uint64_t id = 0;
            const auto [ptr, ec] =
                std::from_chars(token.data() + 1, token.data() + token.size(), id);
            if (ec == std::errc() && ptr == token.data() + token.size()) {
                const std::string* mapped = lookup(id);
                if (!mapped) {
                    throw std::runtime_error("Undefined SPEF *NAME_MAP id: " + std::string(token));
                }
                return *mapped;
            }
        }
        return std::string(token);
    }

    // Expand SPEF node token. If delimiter is present (e.g. "*281152:Y" or
    // "inst/sub:Y"), returns "<expanded_prefix>:<suffix>" using ':' so instance
    // pins match PyPlaceDB's "inst:pin" naming.
    std::string expandNode(std::string_view token, char delimiter, bool* hasSuffix = nullptr) const {
        if (token.size() > 1 && token[0] == '*' &&
            std::isdigit(static_cast<unsigned char>(token[1]))) {
            uint64_t id = 0;
            const char* begin = token.data() + 1;
            const char* end = token.data() + token.size();
            const auto [ptr, ec] = std::from_chars(begin, end, id);
            if (ec == std::errc()) {
                if (ptr == end) {
                    if (hasSuffix) *hasSuffix = false;
                    const std::string* mapped = lookup(id);
                    if (!mapped) {
                        throw std::runtime_error("Undefined SPEF *NAME_MAP id: " +
                                                 std::string(token));
                    }
                    return *mapped;
                }
                if (*ptr == delimiter) {
                    if (hasSuffix) *hasSuffix = true;
                    const std::string* mapped = lookup(id);
                    if (!mapped) {
                        throw std::runtime_error("Undefined SPEF *NAME_MAP id in node: " +
                                                 std::string(token));
                    }
                    std::string out;
                    out.reserve(mapped->size() + 1 + static_cast<size_t>(end - ptr - 1));
                    out.append(*mapped);
                    out.push_back(':');
                    out.append(ptr + 1, static_cast<size_t>(end - ptr - 1));
                    return out;
                }
            }
        }
        // Unmapped identifier: find last unescaped delimiter.
        size_t splitPos = std::string_view::npos;
        for (size_t i = 0; i < token.size(); ++i) {
            if (token[i] == '\\' && i + 1 < token.size()) {
                ++i;
                continue;
            }
            if (token[i] == delimiter) {
                splitPos = i;
            }
        }
        if (splitPos == std::string_view::npos) {
            if (hasSuffix) *hasSuffix = false;
            return std::string(token);
        }
        if (hasSuffix) *hasSuffix = true;
        std::string out(token);
        out[splitPos] = ':';
        return out;
    }

private:
    std::vector<std::string> dense_;
    std::unordered_map<uint64_t, std::string> sparse_;
};

struct TokenView {
    std::string_view text;
    bool quoted = false;
};

void tokenizeLine(std::string_view line, std::vector<TokenView>& tokens) {
    tokens.clear();
    const size_t n = line.size();
    size_t i = 0;
    while (i < n) {
        while (i < n && std::isspace(static_cast<unsigned char>(line[i]))) {
            ++i;
        }
        if (i >= n) break;
        if (i + 1 < n && line[i] == '/' && line[i + 1] == '/') {
            break;  // Line comment.
        }
        if (line[i] == '"') {
            const size_t start = ++i;
            while (i < n && line[i] != '"') {
                if (line[i] == '\\' && i + 1 < n) ++i;
                ++i;
            }
            tokens.push_back({line.substr(start, i - start), true});
            if (i < n && line[i] == '"') ++i;
            continue;
        }
        const size_t start = i;
        while (i < n && !std::isspace(static_cast<unsigned char>(line[i]))) {
            if (line[i] == '\\' && i + 1 < n) {
                i += 2;
                continue;
            }
            if (i + 1 < n && line[i] == '/' && line[i + 1] == '/') {
                break;
            }
            ++i;
        }
        tokens.push_back({line.substr(start, i - start), false});
    }
}

bool parseDouble(std::string_view token, double& out) {
    std::string buf(token);
    char* end = nullptr;
    out = std::strtod(buf.c_str(), &end);
    return end != buf.c_str() && *end == '\0' && std::isfinite(out);
}

double resolveResistanceScaleToOhm(double factor, std::string_view unit) {
    const std::string upper = toUpper(unit);
    if (upper == "OHM") return factor;
    if (upper == "KOHM") return factor * 1e3;
    if (upper == "MOHM") return factor * 1e6;
    throw std::runtime_error("Unsupported SPEF *R_UNIT: " + std::string(unit));
}

double resolveCapacitanceScaleToPf(double factor, std::string_view unit) {
    const std::string upper = toUpper(unit);
    if (upper == "PF") return factor;
    if (upper == "FF") return factor * 1e-3;
    if (upper == "AF") return factor * 1e-6;
    if (upper == "NF") return factor * 1e3;
    throw std::runtime_error("Unsupported SPEF *C_UNIT: " + std::string(unit));
}

class GzLineReader {
public:
    explicit GzLineReader(const std::string& path) {
        file_ = gzopen(path.c_str(), "rb");
        if (!file_) {
            throw std::runtime_error("Failed to open SPEF file: " + path);
        }
        gzbuffer(file_, 1 << 20);
    }

    ~GzLineReader() {
        if (file_) {
            gzclose(file_);
        }
    }

    GzLineReader(const GzLineReader&) = delete;
    GzLineReader& operator=(const GzLineReader&) = delete;

    bool nextLine(std::string& line) {
        line.clear();
        char chunk[65536];
        while (true) {
            char* got = gzgets(file_, chunk, sizeof(chunk));
            if (!got) {
                int errnum = Z_OK;
                const char* msg = gzerror(file_, &errnum);
                if (errnum != Z_OK && errnum != Z_STREAM_END) {
                    throw std::runtime_error(std::string("SPEF gzip read error: ") +
                                             (msg ? msg : "unknown"));
                }
                return !line.empty();
            }
            line.append(chunk);
            if (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
                while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
                    line.pop_back();
                }
                return true;
            }
        }
    }

private:
    gzFile file_ = nullptr;
};

struct LocalNode {
    int32_t pyPin = -1;
    bool hasCoord = false;
    bool isSpefDriver = false;
    double x = 0.0;
    double y = 0.0;
    double groundedCap = 0.0;
    double coupledCap = 0.0;
};

}  // namespace

py::dict prepareSpefTree(const std::string& spefPath,
                         const py::dict& pinNames,
                         const py::dict& netNames,
                         IndexArray pyPinNetArray,
                         IndexArray driverArray,
                         IndexArray eligibleArray,
                         double couplingCapScale) {
    if (!std::isfinite(couplingCapScale) || couplingCapScale < 0.0) {
        throw std::runtime_error("SPEF coupling_cap_scale must be finite and nonnegative");
    }
    const auto pyPinNet = toVector(pyPinNetArray);
    const auto drivers = toVector(driverArray);
    const auto eligible = toVector(eligibleArray);
    const int32_t numNets = static_cast<int32_t>(eligible.size());
    const int32_t numPins = static_cast<int32_t>(pyPinNet.size());
    if (static_cast<int32_t>(drivers.size()) != numNets) {
        throw std::runtime_error("SPEF driver array size does not match eligible net mask");
    }
    for (int32_t p = 0; p < numPins; ++p) {
        if (pyPinNet[p] < 0 || pyPinNet[p] >= numNets) {
            throw std::runtime_error("PyDB pin net out of range");
        }
    }

    std::vector<std::vector<int32_t>> expectedNetPins(numNets);
    for (int32_t p = 0; p < numPins; ++p) {
        expectedNetPins[pyPinNet[p]].push_back(p);
    }

    const NameLookup pinLookup(pinNames, "pin");
    const NameLookup netLookup(netNames, "net");
    SpefNameMap nameMap;

    std::string spefStandard;
    std::string spefDesign;
    char delimiter = ':';
    double rScale = 1.0;
    double cScale = 1e-3;  // IEEE default fallback if omitted, validated when header parsed.
    bool hasRUnit = false;
    bool hasCUnit = false;

    std::vector<int32_t> pinMap(numPins, -1);
    std::vector<int32_t> vertexNet;
    std::vector<int32_t> vertexLayer;
    std::vector<int32_t> parent;
    std::vector<int32_t> topo;
    std::vector<int32_t> topoStart{0};
    std::vector<int32_t> roots;
    std::vector<int32_t> netIds;
    std::vector<int32_t> edgeParent;
    std::vector<int32_t> edgeChild;
    std::vector<double> xy;
    std::vector<double> incomingR;
    std::vector<double> wireCap;
    std::vector<double> edgeR;
    std::vector<double> edgeC;

    std::vector<uint8_t> seenSpefNet(numNets, 0);
    int64_t filteredSpefNets = 0;
    int64_t unmappedSpefNets = 0;
    int64_t incompleteSpefNets = 0;
    int64_t lumpedOnlyNets = 0;
    int64_t parallelMerges = 0;
    int64_t seriesParallelReductions = 0;
    int64_t loopCount = 0;
    int64_t verticesWithCoords = 0;
    double totalGroundedCapPf = 0.0;
    double totalCoupledCapPf = 0.0;
    py::list loopEdgesSample;

    // Per-D_NET reusable scratch buffers.
    int32_t currentNet = -1;
    std::string currentNetCanonical;
    double currentNetLumpedCapPf = 0.0;
    bool skipCurrentNet = true;
    enum class Section { kHeader, kNameMap, kPorts, kConn, kCap, kRes, kOther };
    Section section = Section::kHeader;

    std::vector<LocalNode> localNodes;
    std::unordered_map<std::string, int32_t> localNodeIds;
    std::vector<std::unordered_map<int32_t, double>> adjConductance;

    auto ensureLocalNode = [&](std::string_view rawToken, bool isPortToken = false) -> int32_t {
        if (!isPortToken) {
            const std::string portKey = nameMap.expandNetOrPort(rawToken);
            if (auto portIt = localNodeIds.find(portKey); portIt != localNodeIds.end()) {
                return portIt->second;
            }
        }
        std::string key = isPortToken ? nameMap.expandNetOrPort(rawToken)
                                      : nameMap.expandNode(rawToken, delimiter);
        auto [it, inserted] = localNodeIds.emplace(std::move(key), static_cast<int32_t>(localNodes.size()));
        if (inserted) {
            localNodes.push_back(LocalNode{});
            adjConductance.emplace_back();
        }
        return it->second;
    };

    auto resolveCurrentNetNode = [&](std::string_view rawToken) -> int32_t {
        bool hasSuffix = false;
        const std::string nodeKey = nameMap.expandNode(rawToken, delimiter, &hasSuffix);
        if (auto it = localNodeIds.find(nodeKey); it != localNodeIds.end()) {
            return it->second;
        }
        const std::string portKey = nameMap.expandNetOrPort(rawToken);
        if (auto it = localNodeIds.find(portKey); it != localNodeIds.end()) {
            return it->second;
        }
        const int32_t pyPin = pinLookup.find(nodeKey);
        if (pyPin >= 0 && pyPin < numPins) {
            return pyPinNet[pyPin] == currentNet ? ensureLocalNode(rawToken, false) : -1;
        }
        const int32_t portPin = pinLookup.find(portKey);
        if (portPin >= 0 && portPin < numPins) {
            return pyPinNet[portPin] == currentNet ? ensureLocalNode(rawToken, true) : -1;
        }
        if (hasSuffix) {
            const size_t sep = nodeKey.rfind(':');
            if (sep != std::string::npos &&
                unescapeSpefIdent(std::string_view(nodeKey).substr(0, sep)) == currentNetCanonical) {
                return ensureLocalNode(rawToken, false);
            }
        }
        return -1;
    };

    auto addResistor = [&](int32_t u, int32_t v, double resistanceOhm) {
        if (u == v) return;
        const double r = std::max(0.0, resistanceOhm);
        const double g = 1.0 / std::max(r, kMinConductanceOhm);
        auto [it, inserted] = adjConductance[u].emplace(v, g);
        if (!inserted) {
            it->second += g;
            adjConductance[v][u] += g;
            ++parallelMerges;
        } else {
            adjConductance[v].emplace(u, g);
        }
    };

    auto finalizeCurrentNet = [&]() {
        if (skipCurrentNet || currentNet < 0) {
            currentNet = -1;
            skipCurrentNet = true;
            return;
        }
        const int32_t target = currentNet;
        currentNet = -1;
        skipCurrentNet = true;

        const auto& requiredPins = expectedNetPins[target];
        if (requiredPins.empty()) {
            ++incompleteSpefNets;
            return;
        }

        // Bind all PyDB pins of this net if not already bound in *CONN (some SPEFs
        // mention pins only in *CAP/*RES without a *CONN entry).
        for (auto& [name, localId] : localNodeIds) {
            if (localNodes[localId].pyPin < 0) {
                const int32_t pyPin = pinLookup.find(name);
                if (pyPin >= 0 && pyPin < numPins && pyPinNet[pyPin] == target) {
                    localNodes[localId].pyPin = pyPin;
                }
            }
        }

        std::vector<int32_t> pinToLocal(requiredPins.size(), -1);
        bool allPinsPresent = true;
        for (size_t i = 0; i < requiredPins.size(); ++i) {
            const int32_t reqPin = requiredPins[i];
            int32_t foundLocal = -1;
            for (size_t u = 0; u < localNodes.size(); ++u) {
                if (localNodes[u].pyPin == reqPin) {
                    if (foundLocal >= 0) {
                        // Duplicate local alias for same pin: short them with minimal R.
                        addResistor(foundLocal, static_cast<int32_t>(u), 0.0);
                    } else {
                        foundLocal = static_cast<int32_t>(u);
                    }
                }
            }
            if (foundLocal < 0) {
                allPinsPresent = false;
                break;
            }
            pinToLocal[i] = foundLocal;
        }
        if (!allPinsPresent) {
            ++incompleteSpefNets;
            return;
        }

        // Resolve canonical root (driver) local node.
        int32_t rootLocal = -1;
        const int32_t pyDriver = drivers[target];
        if (pyDriver >= 0) {
            for (size_t i = 0; i < requiredPins.size(); ++i) {
                if (requiredPins[i] == pyDriver) {
                    rootLocal = pinToLocal[i];
                    break;
                }
            }
        }
        if (rootLocal < 0) {
            for (size_t u = 0; u < localNodes.size(); ++u) {
                if (localNodes[u].isSpefDriver && localNodes[u].pyPin >= 0) {
                    rootLocal = static_cast<int32_t>(u);
                    break;
                }
            }
        }
        if (rootLocal < 0) {
            ++incompleteSpefNets;
            return;
        }

        const int32_t localCount = static_cast<int32_t>(localNodes.size());
        std::vector<double> nodeCapPf(localCount, 0.0);
        double sumExplicitCap = 0.0;
        for (int32_t u = 0; u < localCount; ++u) {
            const double gCap = localNodes[u].groundedCap;
            const double cCap = localNodes[u].coupledCap;
            const double total = std::max(0.0, gCap + cCap);
            nodeCapPf[u] = total;
            sumExplicitCap += total;
            totalGroundedCapPf += std::max(0.0, gCap);
            totalCoupledCapPf += std::max(0.0, cCap);
        }

        // Handle lumped-only 1-pin net (e.g., dangling driver output load with 0 resistors).
        bool hasAnyEdge = false;
        for (int32_t u = 0; u < localCount; ++u) {
            if (!adjConductance[u].empty()) {
                hasAnyEdge = true;
                break;
            }
        }
        if (!hasAnyEdge) {
            if (requiredPins.size() == 1) {
                const double cap = sumExplicitCap > 0.0
                                       ? sumExplicitCap
                                       : std::max(0.0, currentNetLumpedCapPf);
                if (parent.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                    throw std::runtime_error("SPEF electrical vertices exceed int32 capacity");
                }
                const int32_t vId = static_cast<int32_t>(parent.size());
                vertexNet.push_back(target);
                vertexLayer.push_back(-1);
                xy.push_back(localNodes[rootLocal].x);
                xy.push_back(localNodes[rootLocal].y);
                if (localNodes[rootLocal].hasCoord) ++verticesWithCoords;
                parent.push_back(-1);
                incomingR.push_back(0.0);
                wireCap.push_back(cap);
                roots.push_back(vId);
                netIds.push_back(target);
                topo.push_back(vId);
                topoStart.push_back(static_cast<int32_t>(topo.size()));
                pinMap[requiredPins[0]] = vId;
                ++lumpedOnlyNets;
                return;
            }
            ++incompleteSpefNets;
            return;
        }

        // Exact Series-Parallel Ladder Reduction:
        // Eliminate non-pin, non-root degree-2 vertices w whose neighbors (a, b)
        // ALREADY share an edge (e.g. parallel multi-via ladders in StarRC),
        // preserving exact parallel conductance and Pi-model Elmore capacitance.
        std::vector<uint8_t> alive(localCount, 1);
        std::queue<int32_t> candidates;
        for (int32_t w = 0; w < localCount; ++w) {
            if (w != rootLocal && localNodes[w].pyPin < 0 && adjConductance[w].size() == 2) {
                candidates.push(w);
            }
        }
        while (!candidates.empty()) {
            const int32_t w = candidates.front();
            candidates.pop();
            if (!alive[w] || w == rootLocal || localNodes[w].pyPin >= 0 ||
                adjConductance[w].size() != 2) {
                continue;
            }
            auto it = adjConductance[w].begin();
            const int32_t a = it->first;
            const double gAw = it->second;
            ++it;
            const int32_t b = it->first;
            const double gBw = it->second;
            // Only contract when (a, b) already has a parallel edge so straight
            // multi-segment wire lines keep their distributed RC vertices intact.
            if (adjConductance[a].find(b) == adjConductance[a].end()) {
                continue;
            }
            const double rAw = 1.0 / gAw;
            const double rBw = 1.0 / gBw;
            const double rSum = rAw + rBw;
            const double gSeries = 1.0 / std::max(rSum, kMinConductanceOhm);
            const double wShareA = rSum > 0.0 ? (rBw / rSum) : 0.5;
            const double wShareB = 1.0 - wShareA;
            nodeCapPf[a] += nodeCapPf[w] * wShareA;
            nodeCapPf[b] += nodeCapPf[w] * wShareB;
            nodeCapPf[w] = 0.0;

            adjConductance[a].erase(w);
            adjConductance[b].erase(w);
            adjConductance[w].clear();
            alive[w] = 0;

            adjConductance[a][b] += gSeries;
            adjConductance[b][a] += gSeries;
            ++seriesParallelReductions;

            for (int32_t nb : {a, b}) {
                if (alive[nb] && nb != rootLocal && localNodes[nb].pyPin < 0 &&
                    adjConductance[nb].size() == 2) {
                    candidates.push(nb);
                }
            }
        }

        // Dijkstra shortest-resistance spanning tree from rootLocal.
        constexpr double kInf = std::numeric_limits<double>::infinity();
        std::vector<double> dist(localCount, kInf);
        std::vector<int32_t> localParent(localCount, -2);
        std::vector<double> localIncomingR(localCount, 0.0);

        using QueueEntry = std::pair<double, int32_t>;
        std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> pq;
        dist[rootLocal] = 0.0;
        localParent[rootLocal] = -1;
        pq.emplace(0.0, rootLocal);

        while (!pq.empty()) {
            const auto [d, u] = pq.top();
            pq.pop();
            if (d > dist[u]) continue;
            for (const auto& [v, g] : adjConductance[u]) {
                if (!alive[v]) continue;
                const double rEdge = g >= (1.0 / kMinConductanceOhm) ? 0.0 : (1.0 / g);
                const double cand = d + rEdge;
                if (cand < dist[v]) {
                    dist[v] = cand;
                    localParent[v] = u;
                    localIncomingR[v] = rEdge;
                    pq.emplace(cand, v);
                }
            }
        }

        // Verify every required PyDB pin on this net was reached from the driver!
        for (size_t i = 0; i < requiredPins.size(); ++i) {
            if (localParent[pinToLocal[i]] == -2) {
                ++incompleteSpefNets;
                return;
            }
        }

        std::vector<std::vector<int32_t>> localChildren(localCount);
        for (int32_t v = 0; v < localCount; ++v) {
            const int32_t p = localParent[v];
            if (p >= 0) {
                localChildren[p].push_back(v);
            }
        }
        std::vector<int32_t> order;
        order.reserve(localCount);
        std::vector<int32_t> dfsStack{rootLocal};
        while (!dfsStack.empty()) {
            const int32_t u = dfsStack.back();
            dfsStack.pop_back();
            order.push_back(u);
            const auto& ch = localChildren[u];
            for (auto it = ch.rbegin(); it != ch.rend(); ++it) {
                dfsStack.push_back(*it);
            }
        }

        // Fold any unreachable floating parasitic-only islands' capacitance into
        // root? No: floating disconnected islands do not load the driver. Only
        // reached vertices enter the electrical tree. Record residual non-tree
        // loop edges among reached vertices.
        for (int32_t u : order) {
            for (const auto& [v, g] : adjConductance[u]) {
                if (u < v && localParent[v] != -2 && localParent[u] != v && localParent[v] != u) {
                    ++loopCount;
                    if (loopEdgesSample.size() < kMaxLoopSample) {
                        py::dict ev;
                        ev["net_id"] = target;
                        ev["resistance_ohm"] = g >= (1.0 / kMinConductanceOhm) ? 0.0 : (1.0 / g);
                        loopEdgesSample.append(ev);
                    }
                }
            }
        }

        if (parent.size() + order.size() >=
            static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("SPEF electrical vertices exceed int32 capacity");
        }

        std::vector<int32_t> localToGlobal(localCount, -1);
        for (int32_t u : order) {
            const int32_t gId = static_cast<int32_t>(parent.size());
            localToGlobal[u] = gId;
            vertexNet.push_back(target);
            vertexLayer.push_back(-1);
            xy.push_back(localNodes[u].x);
            xy.push_back(localNodes[u].y);
            if (localNodes[u].hasCoord) ++verticesWithCoords;
            const int32_t pLocal = localParent[u];
            const int32_t pGlobal = pLocal < 0 ? -1 : localToGlobal[pLocal];
            parent.push_back(pGlobal);
            incomingR.push_back(pGlobal < 0 ? 0.0 : localIncomingR[u]);
            wireCap.push_back(nodeCapPf[u]);
            topo.push_back(gId);
            if (pGlobal >= 0) {
                edgeParent.push_back(pGlobal);
                edgeChild.push_back(gId);
                edgeR.push_back(localIncomingR[u]);
                edgeC.push_back(0.0);
            }
        }

        roots.push_back(localToGlobal[rootLocal]);
        netIds.push_back(target);
        topoStart.push_back(static_cast<int32_t>(topo.size()));
        for (size_t i = 0; i < requiredPins.size(); ++i) {
            pinMap[requiredPins[i]] = localToGlobal[pinToLocal[i]];
        }
    };

    GzLineReader reader(spefPath);
    std::string line;
    std::vector<TokenView> tokens;
    tokens.reserve(16);

    while (reader.nextLine(line)) {
        tokenizeLine(line, tokens);
        if (tokens.empty()) continue;

        const std::string_view head = tokens[0].text;
        if (!tokens[0].quoted && !head.empty() && head[0] == '*') {
            if (head == "*D_NET") {
                finalizeCurrentNet();
                section = Section::kOther;
                if (tokens.size() < 3) {
                    throw std::runtime_error("Malformed SPEF *D_NET header");
                }
                if (!hasRUnit || !hasCUnit) {
                    throw std::runtime_error("SPEF missing *R_UNIT or *C_UNIT before *D_NET");
                }
                const std::string netName = nameMap.expandNetOrPort(tokens[1].text);
                double lcap = 0.0;
                if (!parseDouble(tokens[2].text, lcap)) {
                    throw std::runtime_error("Invalid lumped cap on *D_NET: " + netName);
                }
                currentNetLumpedCapPf = lcap * cScale;
                const int32_t target = netLookup.find(netName);
                if (target < 0 || target >= numNets) {
                    ++unmappedSpefNets;
                    skipCurrentNet = true;
                    continue;
                }
                seenSpefNet[target] = 1;
                if (!eligible[target]) {
                    ++filteredSpefNets;
                    skipCurrentNet = true;
                    continue;
                }
                currentNet = target;
                currentNetCanonical = unescapeSpefIdent(netName);
                skipCurrentNet = false;
                localNodes.clear();
                localNodeIds.clear();
                adjConductance.clear();
                continue;
            }
            if (head == "*END") {
                finalizeCurrentNet();
                section = Section::kOther;
                continue;
            }
            if (head == "*NAME_MAP") {
                section = Section::kNameMap;
                continue;
            }
            if (head == "*PORTS") {
                section = Section::kPorts;
                continue;
            }
            if (head == "*CONN") {
                section = Section::kConn;
                continue;
            }
            if (head == "*CAP") {
                section = Section::kCap;
                continue;
            }
            if (head == "*RES") {
                section = Section::kRes;
                continue;
            }
            if (head == "*INDUC") {
                section = Section::kOther;
                continue;
            }

            // Header directives and *CONN entries (*P, *I, *N) also start with '*'.
            if (section == Section::kHeader || section == Section::kNameMap ||
                section == Section::kPorts) {
                if (head == "*SPEF" && tokens.size() >= 2) {
                    spefStandard = std::string(tokens[1].text);
                    section = Section::kHeader;
                    continue;
                }
                if (head == "*DESIGN" && tokens.size() >= 2) {
                    spefDesign = std::string(tokens[1].text);
                    section = Section::kHeader;
                    continue;
                }
                if (head == "*DELIMITER" && tokens.size() >= 2 && !tokens[1].text.empty()) {
                    delimiter = tokens[1].text[0];
                    section = Section::kHeader;
                    continue;
                }
                if (head == "*R_UNIT" && tokens.size() >= 3) {
                    double factor = 0.0;
                    if (!parseDouble(tokens[1].text, factor) || factor <= 0.0) {
                        throw std::runtime_error("Invalid SPEF *R_UNIT factor");
                    }
                    rScale = resolveResistanceScaleToOhm(factor, tokens[2].text);
                    hasRUnit = true;
                    section = Section::kHeader;
                    continue;
                }
                if (head == "*C_UNIT" && tokens.size() >= 3) {
                    double factor = 0.0;
                    if (!parseDouble(tokens[1].text, factor) || factor <= 0.0) {
                        throw std::runtime_error("Invalid SPEF *C_UNIT factor");
                    }
                    cScale = resolveCapacitanceScaleToPf(factor, tokens[2].text);
                    hasCUnit = true;
                    section = Section::kHeader;
                    continue;
                }
                if (section == Section::kNameMap && head.size() > 1 &&
                    std::isdigit(static_cast<unsigned char>(head[1])) && tokens.size() >= 2) {
                    uint64_t id = 0;
                    const auto [ptr, ec] =
                        std::from_chars(head.data() + 1, head.data() + head.size(), id);
                    if (ec != std::errc() || ptr != head.data() + head.size()) {
                        throw std::runtime_error("Invalid *NAME_MAP entry id: " +
                                                 std::string(head));
                    }
                    nameMap.insert(id, std::string(tokens[1].text));
                    continue;
                }
                continue;
            }
        }

        if (skipCurrentNet) continue;

        if (section == Section::kConn) {
            if ((head == "*I" || head == "*P") && tokens.size() >= 3) {
                const bool isPort = (head == "*P");
                const int32_t nodeId = ensureLocalNode(tokens[1].text, isPort);
                LocalNode& node = localNodes[nodeId];
                const std::string pinKey = isPort ? nameMap.expandNetOrPort(tokens[1].text)
                                                  : nameMap.expandNode(tokens[1].text, delimiter);
                const int32_t pyPin = pinLookup.find(pinKey);
                if (pyPin >= 0 && pyPin < numPins && pyPinNet[pyPin] == currentNet) {
                    node.pyPin = pyPin;
                }
                const std::string_view dir = tokens[2].text;
                if ((!isPort && (dir == "O" || dir == "B")) ||
                    (isPort && (dir == "I" || dir == "B"))) {
                    node.isSpefDriver = true;
                }
                for (size_t i = 3; i < tokens.size(); ++i) {
                    if (tokens[i].text == "*C" && i + 2 < tokens.size()) {
                        double px = 0.0, pyVal = 0.0;
                        if (parseDouble(tokens[i + 1].text, px) &&
                            parseDouble(tokens[i + 2].text, pyVal)) {
                            node.x = px;
                            node.y = pyVal;
                            node.hasCoord = true;
                        }
                        i += 2;
                    } else if (tokens[i].text == "*D") {
                        node.isSpefDriver = true;
                    }
                }
            } else if (head == "*N" && tokens.size() >= 2) {
                const int32_t nodeId = ensureLocalNode(tokens[1].text, false);
                LocalNode& node = localNodes[nodeId];
                for (size_t i = 2; i + 2 < tokens.size(); ++i) {
                    if (tokens[i].text == "*C") {
                        double px = 0.0, pyVal = 0.0;
                        if (parseDouble(tokens[i + 1].text, px) &&
                            parseDouble(tokens[i + 2].text, pyVal)) {
                            node.x = px;
                            node.y = pyVal;
                            node.hasCoord = true;
                        }
                        break;
                    }
                }
            }
        } else if (section == Section::kCap) {
            if (tokens.size() == 3) {
                // Grounded capacitor: <id> <node> <val>
                double val = 0.0;
                if (!parseDouble(tokens[2].text, val)) {
                    throw std::runtime_error("Invalid SPEF grounded *CAP value");
                }
                const int32_t u = ensureLocalNode(tokens[1].text);
                localNodes[u].groundedCap += val * cScale;
            } else if (tokens.size() >= 4) {
                // Coupling capacitor: <id> <node1> <node2> <val>
                double val = 0.0;
                if (!parseDouble(tokens[3].text, val)) {
                    throw std::runtime_error("Invalid SPEF coupling *CAP value");
                }
                const double scaled = val * cScale * couplingCapScale;
                // In StarRC / IEEE 1481 D_NET blocks, node1 belongs to the current
                // D_NET while node2 is typically on an aggressor net (which has its
                // own symmetric D_NET entry). Never create phantom vertices for
                // external aggressor nodes inside this D_NET!
                const int32_t u1 = resolveCurrentNetNode(tokens[1].text);
                const int32_t u2 = resolveCurrentNetNode(tokens[2].text);
                if (u1 >= 0 && u2 >= 0) {
                    // Intra-net coupling between two nodes on the same net:
                    // match ReadParasitics.cpp convention.
                    localNodes[u1].coupledCap += scaled;
                    localNodes[u2].coupledCap += scaled;
                } else if (u1 >= 0) {
                    localNodes[u1].coupledCap += scaled;
                } else if (u2 >= 0) {
                    localNodes[u2].coupledCap += scaled;
                }
            }
        } else if (section == Section::kRes) {
            if (tokens.size() >= 4) {
                double val = 0.0;
                if (!parseDouble(tokens[3].text, val)) {
                    throw std::runtime_error("Invalid SPEF *RES value");
                }
                const int32_t u = ensureLocalNode(tokens[1].text);
                const int32_t v = ensureLocalNode(tokens[2].text);
                addResistor(u, v, val * rScale);
            }
        }
    }
    finalizeCurrentNet();

    int64_t missingSpefNets = 0;
    for (int32_t n = 0; n < numNets; ++n) {
        if (eligible[n] && !seenSpefNet[n]) {
            ++missingSpefNets;
        }
    }
    const int32_t coveredNetCount = static_cast<int32_t>(netIds.size());
    const int32_t totalFilteredNets = numNets - coveredNetCount;

    std::vector<int32_t> childStart(parent.size() + 1, 0);
    std::vector<int32_t> children(edgeChild.size());
    for (int32_t p : edgeParent) {
        ++childStart[static_cast<size_t>(p) + 1];
    }
    for (size_t i = 1; i < childStart.size(); ++i) {
        childStart[i] += childStart[i - 1];
    }
    auto cursor = childStart;
    for (size_t i = 0; i < edgeChild.size(); ++i) {
        children[static_cast<size_t>(cursor[static_cast<size_t>(edgeParent[i])]++)] = edgeChild[i];
    }

    py::dict result;
    result["pin_to_vertex"] = toArray(pinMap);
    result["pin_grid_vertex"] = toArray(pinMap);
    result["vertex_net"] = toArray(vertexNet);
    result["vertex_layer"] = toArray(vertexLayer);
    auto coordinates =
        py::array_t<double>({static_cast<py::ssize_t>(parent.size()), py::ssize_t(2)});
    std::copy(xy.begin(), xy.end(), coordinates.mutable_data());
    result["vertex_xy_um"] = coordinates;
    result["parent"] = toArray(parent);
    result["incoming_resistance"] = toArray(incomingR);
    result["wire_cap"] = toArray(wireCap);
    result["child_start"] = toArray(childStart);
    result["child_vertex"] = toArray(children);
    result["topo_order"] = toArray(topo);
    result["net_topo_start"] = toArray(topoStart);
    result["root_vertex"] = toArray(roots);
    result["net_ids"] = toArray(netIds);
    result["edge_parent"] = toArray(edgeParent);
    result["edge_child"] = toArray(edgeChild);
    result["edge_resistance"] = toArray(edgeR);
    result["edge_capacitance"] = toArray(edgeC);
    result["filtered_net_count"] = totalFilteredNets;

    py::dict model;
    model["spef_standard"] = spefStandard;
    model["spef_design"] = spefDesign;
    model["r_unit_scale_ohm"] = rScale;
    model["c_unit_scale_pf"] = cScale;
    model["coupling_cap_scale"] = couplingCapScale;
    model["grounded_wire_cap_pf"] = totalGroundedCapPf;
    model["coupled_wire_cap_pf"] = totalCoupledCapPf;
    model["parallel_resistor_merges"] = parallelMerges;
    model["series_parallel_reductions"] = seriesParallelReductions;
    model["electrical_reduction_model"] =
        "series_parallel_ladder_plus_dijkstra_shortest_resistance_tree";
    model["loop_count"] = loopCount;
    model["loop_edges"] = loopEdgesSample;
    model["filtered_ineligible_spef_net_count"] = filteredSpefNets;
    model["unmapped_spef_net_count"] = unmappedSpefNets;
    model["missing_spef_net_count"] = missingSpefNets;
    model["incomplete_spef_net_count"] = incompleteSpefNets;
    model["lumped_only_net_count"] = lumpedOnlyNets;
    model["vertices_with_coordinates"] = verticesWithCoords;
    result["model_report"] = model;
    return result;
}

}  // namespace gr_parasitics

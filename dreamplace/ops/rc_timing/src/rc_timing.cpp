#include <torch/extension.h>
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <numeric> 
#include <algorithm> 
#include <cmath> 

// RC树节点结构
struct RCTreeNode {
    int id;                               // 节点ID
    float x, y;                           // 坐标
    float pin_cap;                        // 引脚电容
    int parent;                           // 父节点ID，根为-1
    std::vector<int> children;            // 子节点索引
    std::vector<float> resistance;        // 连接子节点的电阻
    std::vector<float> wire_cap;          // 连接线的电容
};

// 论文中Elmore模型的节点信息
struct ElmoreNodeInfo {
    int global_id;           // 全局节点ID
    float pin_cap;           // 节点电容
    bool is_sink;            // 是否为sink节点
    int output_idx;          // 输出索引(仅对sink有效)
    
    float load;              // Load(u)
    float delay;             // Delay(u) 
    float ldelay;            // LDelay(u) 
    float beta;              // Beta(u) 
    float impulse;           // Impulse(u) 
};

// 从斯坦纳树构建RC树并计算Elmore Delay

torch::Tensor forward(
    torch::Tensor pos,               // 所有节点坐标 [num_pins*2]
    torch::Tensor branch_u,          // 斯坦纳树边的起点 [num_edges]
    torch::Tensor branch_v,          // 斯坦纳树边的终点 [num_edges]
    torch::Tensor net_branch_start,  // 每个网络边的起始索引 [num_nets+1]
    torch::Tensor pin_caps,          // 每个引脚的电容 [num_pins]
    torch::Tensor driver_pin_indices, // 每个网络的驱动引脚索引 [num_nets]
    torch::Tensor sink_pin_indices,   // 网络中的sink pin索引 [num_sinks]
    torch::Tensor net_sink_start,     // 每个网络的sink pin起始索引 [num_nets+1]
    torch::Tensor wire_widths,        // 线宽 [num_edges]
    torch::Tensor res_per_micron,     // 每微米电阻 [1]
    torch::Tensor cap_per_micron,     // 每微米电容 [1]
    torch::Tensor edge_cap_per_micron, // 边缘电容 [1]
    int ignore_net_degree            // 忽略网络的度数阈值
) {
    // --- 输入检查 ---
    TORCH_CHECK(pos.dim() == 1 && pos.size(0) % 2 == 0, "Position tensor must be [num_pins*2]");
    TORCH_CHECK(branch_u.dim() == 1, "Branch_u must be [num_edges]");
    TORCH_CHECK(branch_v.dim() == 1, "Branch_v must be [num_edges]");
    TORCH_CHECK(branch_u.size(0) == branch_v.size(0), "Branch_u and branch_v must have the same size");
    TORCH_CHECK(net_branch_start.dim() == 1, "Net branch start must be [num_nets+1]");
    TORCH_CHECK(pin_caps.dim() == 1, "Pin capacitance must be [num_pins]");
    TORCH_CHECK(driver_pin_indices.dim() == 1, "Driver pin indices must be [num_nets]");
    TORCH_CHECK(sink_pin_indices.dim() == 1, "Sink pin indices must be [num_sinks]");
    TORCH_CHECK(net_sink_start.dim() == 1 && net_sink_start.size(0) == net_branch_start.size(0), "Net sink start shape error");
    TORCH_CHECK(wire_widths.dim() == 1 && wire_widths.size(0) == branch_u.size(0), "Wire widths must be [num_edges]");

    // --- 参数设置 ---
    int num_nets = net_branch_start.size(0) - 1;
    int num_pins = pos.size(0) / 2;
    int num_edges = branch_u.size(0);
    int num_sinks = sink_pin_indices.size(0);

    auto pos_acc = pos.accessor<float, 1>();
    auto branch_u_acc = branch_u.accessor<int, 1>();
    auto branch_v_acc = branch_v.accessor<int, 1>();
    auto net_branch_start_acc = net_branch_start.accessor<int, 1>();
    auto cap_acc = pin_caps.accessor<float, 1>();
    auto driver_acc = driver_pin_indices.accessor<int, 1>();
    auto sink_acc = sink_pin_indices.accessor<int, 1>();
    auto net_sink_start_acc = net_sink_start.accessor<int, 1>();
    auto width_acc = wire_widths.accessor<float, 1>();
    auto res_per_micron_acc = res_per_micron.accessor<float, 1>();
    auto cap_per_micron_acc = cap_per_micron.accessor<float, 1>();
    auto edge_cap_per_micron_acc = edge_cap_per_micron.accessor<float, 1>();

    // --- 输出张量: 延迟和impulse ---
    auto elmore_delays = torch::zeros({num_sinks}, pos.options());
    auto impulse_responses = torch::zeros({num_sinks}, pos.options());
    auto delay_acc = elmore_delays.accessor<float, 1>();
    auto impulse_acc = impulse_responses.accessor<float, 1>();

    int current_sink_offset = 0;

    // --- 处理每个网络 ---
    for (int net_idx = 0; net_idx < num_nets; ++net_idx) {
        int net_pin_count = net_branch_start_acc[net_idx + 1] - net_branch_start_acc[net_idx];
        int sink_count_in_net = net_sink_start_acc[net_idx + 1] - net_sink_start_acc[net_idx];

        if (net_pin_count > ignore_net_degree) {
            current_sink_offset += sink_count_in_net;
            continue;
        }

        int edge_start = net_branch_start_acc[net_idx];
        int edge_end = net_branch_start_acc[net_idx + 1];

        if (edge_start == edge_end || sink_count_in_net == 0) {
             current_sink_offset += sink_count_in_net;
             continue;
        }

        // 1. 收集节点和基本信息
        std::unordered_map<int, int> global_to_local;
        std::vector<ElmoreNodeInfo> nodes;
        std::vector<int> local_to_global;
        {
            std::unordered_set<int> net_nodes_global_ids;
            for (int i = edge_start; i < edge_end; ++i) {
                if (branch_u_acc[i] >= 0 && branch_u_acc[i] < num_pins) {
                    net_nodes_global_ids.insert(branch_u_acc[i]);
                }
                if (branch_v_acc[i] >= 0 && branch_v_acc[i] < num_pins) {
                     net_nodes_global_ids.insert(branch_v_acc[i]);
                }
            }
            
            int global_driver_id = driver_acc[net_idx];
            if (global_driver_id >= 0 && global_driver_id < num_pins) {
                net_nodes_global_ids.insert(global_driver_id);
            }

            int local_idx = 0;
            for (int node_id : net_nodes_global_ids) {
                global_to_local[node_id] = local_idx;
                local_to_global.push_back(node_id);
                
                nodes.emplace_back();
                nodes.back().global_id = node_id;
                nodes.back().pin_cap = cap_acc[node_id];
                nodes.back().is_sink = false;
                nodes.back().output_idx = -1;
                
                // 初始化Elmore中间值
                nodes.back().load = 0.0f;
                nodes.back().delay = 0.0f;
                nodes.back().ldelay = 0.0f;
                nodes.back().beta = 0.0f;
                nodes.back().impulse = 0.0f;
                
                local_idx++;
            }
        }
        int local_num_nodes = nodes.size();
        if (local_num_nodes == 0) {
            current_sink_offset += sink_count_in_net;
            continue;
        }

        // 标记sink节点
        for(int sink_k = net_sink_start_acc[net_idx]; sink_k < net_sink_start_acc[net_idx+1]; ++sink_k) {
            int global_sink_id = sink_acc[sink_k];
            if (global_sink_id >= 0 && global_sink_id < num_pins && global_to_local.count(global_sink_id)) {
                int local_sink_id = global_to_local[global_sink_id];
                nodes[local_sink_id].is_sink = true;
                nodes[local_sink_id].output_idx = current_sink_offset + (sink_k - net_sink_start_acc[net_idx]);
            }
        }

        // 获取驱动节点ID
        int global_driver = driver_acc[net_idx];
        int local_driver = (global_driver >= 0 && global_driver < num_pins && global_to_local.count(global_driver)) ? global_to_local[global_driver] : -1;

        if (local_driver == -1) {
            current_sink_offset += sink_count_in_net;
            continue;
        }

        // 2. 构建
        std::unordered_map<int, std::vector<int>> adj; // parent->children
        std::unordered_map<int, int> parent_map;       // child->parent
        std::unordered_map<int, float> edge_res;       // child->resistance_to_parent
        std::vector<int> in_degree(local_num_nodes, 0);

        std::queue<int> q;
        std::vector<bool> visited(local_num_nodes, false);

        q.push(local_driver);
        visited[local_driver] = true;
        parent_map[local_driver] = -1; // 根节点无父节点

        while (!q.empty()) {
            int current = q.front();
            q.pop();

            for (int i = edge_start; i < edge_end; ++i) {
                int u_global = branch_u_acc[i];
                int v_global = branch_v_acc[i];

                if (!global_to_local.count(u_global) || !global_to_local.count(v_global)) continue;

                int u_local = global_to_local[u_global];
                int v_local = global_to_local[v_global];

                int child = -1;
                int parent = -1;

                if (visited[u_local] && !visited[v_local]) {
                    parent = u_local;
                    child = v_local;
                } else if (visited[v_local] && !visited[u_local]) {
                    parent = v_local;
                    child = u_local;
                }

                if (parent == current && child != -1) {
                    visited[child] = true;
                    q.push(child);

                    adj[parent].push_back(child);
                    parent_map[child] = parent;
                    in_degree[child]++;

                    // 计算电阻
                    int parent_global = nodes[parent].global_id;
                    int child_global = nodes[child].global_id;

                    float x1 = pos_acc[2 * parent_global];
                    float y1 = pos_acc[2 * parent_global + 1];
                    float x2 = pos_acc[2 * child_global];
                    float y2 = pos_acc[2 * child_global + 1];

                    float length = std::abs(x1 - x2) + std::abs(y1 - y2);
                    float width = width_acc[i];

                    float res_ohm_per_micron = (width > 1e-9) ? (res_per_micron_acc[0] / width) : 0.0f;
                    float resistance = res_ohm_per_micron * length;

                    // 存储电阻
                    edge_res[child] = resistance;
                    
                    // 更新子节点电容 (在RC树中电容被分配到节点上)
                    float area_cap = width * length * cap_per_micron_acc[0];
                    float edge_cap_val = 2 * (length + width) * edge_cap_per_micron_acc[0];
                    float segment_cap = area_cap + edge_cap_val;
                    
                    // 线段电容分配给两端节点 (Pi模型)
                    nodes[parent].pin_cap += segment_cap / 2.0f;
                    nodes[child].pin_cap += segment_cap / 2.0f;
                }
            }
        }

        // 3. 拓扑排序 
        std::vector<int> sorted_nodes;
        std::queue<int> topo_q;

        for (int node_idx = 0; node_idx < local_num_nodes; ++node_idx) {
            if (in_degree[node_idx] == 0) {
                topo_q.push(node_idx);
            }
        }

        if (topo_q.size() != 1 || topo_q.front() != local_driver) {
            current_sink_offset += sink_count_in_net;
            continue;
        }

        while (!topo_q.empty()) {
            int u = topo_q.front();
            topo_q.pop();
            sorted_nodes.push_back(u);

            if (adj.count(u)) {
                for (int v : adj[u]) {
                    in_degree[v]--;
                    if (in_degree[v] == 0) {
                        topo_q.push(v);
                    }
                }
            }
        }

        if (sorted_nodes.size() != local_num_nodes) {
            current_sink_offset += sink_count_in_net;
            continue;
        }

        // 4. Elmore延迟计算 - 4次动态规划传递 (7a-7e)
        
        // Load(u)
        for (auto it = sorted_nodes.rbegin(); it != sorted_nodes.rend(); ++it) {
            int u = *it;
            ElmoreNodeInfo &node = nodes[u];
            
            // Load(u) = Cap(u) + ∑(child v) Load(v)
            node.load = node.pin_cap;
            
            if (adj.count(u)) {
                for (int v : adj[u]) {
                    node.load += nodes[v].load;
                }
            }
        }
        
        // Delay(u)
        for (int u : sorted_nodes) {
            ElmoreNodeInfo &node = nodes[u];
            
            if (parent_map.count(u) && parent_map[u] != -1) {
                int fa_u = parent_map[u];
                // Delay(u) = Delay(fa(u)) + Res(fa(u)→u) · Load(u)
                node.delay = nodes[fa_u].delay + edge_res[u] * node.load;
            } else {
                // 根节点延迟为0
                node.delay = 0.0f;
            }
        }
        
        // LDelay(u)
        for (auto it = sorted_nodes.rbegin(); it != sorted_nodes.rend(); ++it) {
            int u = *it;
            ElmoreNodeInfo &node = nodes[u];
            
            // LDelay(u) = Cap(u) · Delay(u) + ∑(child v) LDelay(v)
            node.ldelay = node.pin_cap * node.delay;
            
            if (adj.count(u)) {
                for (int v : adj[u]) {
                    node.ldelay += nodes[v].ldelay;
                }
            }
        }
        
        // Beta(u) Impulse(u)
        for (int u : sorted_nodes) {
            ElmoreNodeInfo &node = nodes[u];
            
            if (parent_map.count(u) && parent_map[u] != -1) {
                int fa_u = parent_map[u];
                // Beta(u) = Beta(fa(u)) + Res(fa(u)→u) · LDelay(u)
                node.beta = nodes[fa_u].beta + edge_res[u] * node.ldelay;
            } else {
                // 根节点beta为0
                node.beta = 0.0f;
            }
            
            // Impulse(u) = √(2 · Beta(u) - Delay²(u))
            float radicand = 2.0f * node.beta - node.delay * node.delay;
            node.impulse = (radicand > 0) ? std::sqrt(radicand) : 0.0f;
        }

        // 5. 提取sink节点的延迟和impulse响应
        for (int u = 0; u < local_num_nodes; ++u) {
            if (nodes[u].is_sink && nodes[u].output_idx >= 0) {
                delay_acc[nodes[u].output_idx] = nodes[u].delay;
                impulse_acc[nodes[u].output_idx] = nodes[u].impulse;
            }
        }

        current_sink_offset += sink_count_in_net;
    }

    return elmore_delays;
}

// 反向传播 - 占位函数 
torch::Tensor backward(
    torch::Tensor grad_elmore_delay,
    torch::Tensor pos,
    torch::Tensor branch_u,
    torch::Tensor branch_v,
    torch::Tensor net_branch_start,
    torch::Tensor pin_caps,
    torch::Tensor driver_pin_indices,
    torch::Tensor sink_pin_indices,
    torch::Tensor net_sink_start,
    torch::Tensor wire_widths,
    torch::Tensor res_per_micron,
    torch::Tensor cap_per_micron,
    torch::Tensor edge_cap_per_micron
) {
    TORCH_CHECK(false, "Backward pass for Elmore delay calculation is not implemented yet.");
    int num_pins = pos.size(0) / 2;
    auto grad_pos = torch::zeros_like(pos);
    return grad_pos;
}

// PyBind定义
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Calculate Elmore delay using 4-pass DP (forward)",
          py::arg("pos"),
          py::arg("branch_u"),
          py::arg("branch_v"),
          py::arg("net_branch_start"),
          py::arg("pin_caps"),
          py::arg("driver_pin_indices"),
          py::arg("sink_pin_indices"),
          py::arg("net_sink_start"),
          py::arg("wire_widths"),
          py::arg("res_per_micron"),
          py::arg("cap_per_micron"),
          py::arg("edge_cap_per_micron"),
          py::arg("ignore_net_degree"));

    m.def("backward", &backward, "Calculate Elmore delay gradients (backward) - NOT IMPLEMENTED",
          py::arg("grad_elmore_delay"),
          py::arg("pos"),
          py::arg("branch_u"),
          py::arg("branch_v"),
          py::arg("net_branch_start"),
          py::arg("pin_caps"),
          py::arg("driver_pin_indices"),
          py::arg("sink_pin_indices"),
          py::arg("net_sink_start"),
          py::arg("wire_widths"),
          py::arg("res_per_micron"),
          py::arg("cap_per_micron"),
          py::arg("edge_cap_per_micron"));
}

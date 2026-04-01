from dataclasses import dataclass
import logging
import math
import os
from typing import Dict, List, Sequence

import numpy as np
import torch


logger = logging.getLogger(__name__)


@dataclass
class LeidenClusteringResult:
    cluster_ids_by_level: List[torch.Tensor]
    num_clusters_by_level: List[int]
    resolutions_used: List[float]


def _import_leiden_stack():
    try:
        import igraph as ig  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "modularity_inflation_flag=1 requires python-igraph to be installed"
        ) from exc

    try:
        import leidenalg  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "modularity_inflation_flag=1 requires leidenalg to be installed"
        ) from exc

    return ig, leidenalg


def _to_numpy_int(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(values, dtype=np.int64)


def _to_numpy_float(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(values, dtype=np.float64)


def _to_cpu_long_tensor(values) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().to(dtype=torch.int64)
    return torch.as_tensor(values, dtype=torch.int64)


def normalize_cluster_ids(cluster_ids) -> torch.Tensor:
    cluster_ids_np = _to_numpy_int(cluster_ids)
    if cluster_ids_np.size == 0:
        return torch.empty(0, dtype=torch.int64)
    _, inverse = np.unique(cluster_ids_np, return_inverse=True)
    return torch.from_numpy(inverse.astype(np.int64, copy=False))


def _cluster_size_stats(cluster_ids: torch.Tensor) -> str:
    if cluster_ids.numel() == 0:
        return "clusters=0"
    counts = torch.bincount(cluster_ids.cpu())
    return (
        "clusters=%d min=%d max=%d mean=%.2f"
        % (
            int(counts.numel()),
            int(counts.min().item()),
            int(counts.max().item()),
            float(counts.to(dtype=torch.float64).mean().item()),
        )
    )


def build_star_model_graph(
    flat_net2pin_map,
    flat_net2pin_start_map,
    pin2node_map,
    num_movable_nodes,
    num_nets,
    net_weights=None,
    ignore_net_degree=100,
):
    ig, _ = _import_leiden_stack()

    flat_net2pin_map = _to_numpy_int(flat_net2pin_map)
    flat_net2pin_start_map = _to_numpy_int(flat_net2pin_start_map)
    pin2node_map = _to_numpy_int(pin2node_map)
    num_movable_nodes = int(num_movable_nodes)
    num_nets = int(num_nets)

    if net_weights is None:
        net_weights_np = np.ones(num_nets, dtype=np.float64)
    else:
        net_weights_np = _to_numpy_float(net_weights)

    edges = []
    weights = []
    virtual_node_offset = int(num_movable_nodes)
    virtual_node_count = 0
    eligible_nets = 0

    for net_id in range(num_nets):
        start = int(flat_net2pin_start_map[net_id])
        end = int(flat_net2pin_start_map[net_id + 1])
        if end - start < 2:
            continue

        pins = flat_net2pin_map[start:end]
        nodes = pin2node_map[pins]
        movable_nodes = np.unique(nodes[nodes < num_movable_nodes])
        movable_degree = int(movable_nodes.size)
        if movable_degree < 2 or movable_degree > int(ignore_net_degree):
            continue

        eligible_nets += 1
        virtual_node_id = virtual_node_offset + virtual_node_count
        virtual_node_count += 1
        edge_weight = float(net_weights_np[net_id]) / float(max(movable_degree, 1))
        for node_id in movable_nodes:
            edges.append((int(node_id), int(virtual_node_id)))
            weights.append(edge_weight)

    graph = ig.Graph(
        n=int(num_movable_nodes + virtual_node_count),
        edges=edges,
        directed=False,
    )
    if weights:
        graph.es["weight"] = weights

    logger.info(
        "Built modularity star-model graph: movable_nodes=%d eligible_nets=%d virtual_nodes=%d edges=%d",
        num_movable_nodes,
        eligible_nets,
        virtual_node_count,
        len(edges),
    )
    return graph


def run_leiden_multi_resolution(
    graph,
    resolutions,
    num_movable_nodes,
    seed=42,
) -> List[torch.Tensor]:
    _, leidenalg = _import_leiden_stack()

    num_movable_nodes = int(num_movable_nodes)
    resolutions = [float(resolution) for resolution in resolutions]
    if not resolutions:
        resolutions = [1.0]

    if graph.vcount() < num_movable_nodes:
        raise ValueError(
            "Leiden graph has fewer vertices than movable nodes: %d < %d"
            % (graph.vcount(), num_movable_nodes)
        )

    if graph.ecount() == 0:
        singleton_clusters = torch.arange(num_movable_nodes, dtype=torch.int64)
        return [singleton_clusters.clone() for _ in resolutions]

    weights = graph.es["weight"] if "weight" in graph.es.attributes() else None
    cluster_ids_by_level: List[torch.Tensor] = []
    for resolution in resolutions:
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=float(resolution),
            seed=int(seed),
        )
        membership = np.asarray(
            partition.membership[:num_movable_nodes],
            dtype=np.int64,
        )
        cluster_ids = normalize_cluster_ids(membership)
        cluster_ids_by_level.append(cluster_ids)
        logger.info(
            "Leiden clustering at resolution %.4f: %s",
            resolution,
            _cluster_size_stats(cluster_ids),
        )

    return cluster_ids_by_level


class _UnionFind(object):
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, lhs: int, rhs: int) -> None:
        root_lhs = self.find(lhs)
        root_rhs = self.find(rhs)
        if root_lhs == root_rhs:
            return
        if self.rank[root_lhs] < self.rank[root_rhs]:
            root_lhs, root_rhs = root_rhs, root_lhs
        self.parent[root_rhs] = root_lhs
        if self.rank[root_lhs] == self.rank[root_rhs]:
            self.rank[root_lhs] += 1


def _split_members_by_distance(pos_x, pos_y, max_distance):
    num_members = len(pos_x)
    if num_members <= 1:
        return [np.arange(num_members, dtype=np.int64)]

    max_distance = max(float(max_distance), 1e-12)
    inv_bucket = 1.0 / max_distance
    bucket_map: Dict[tuple, List[int]] = {}
    buckets = []
    for idx in range(num_members):
        bucket = (
            int(math.floor(float(pos_x[idx]) * inv_bucket)),
            int(math.floor(float(pos_y[idx]) * inv_bucket)),
        )
        buckets.append(bucket)
        bucket_map.setdefault(bucket, []).append(idx)

    uf = _UnionFind(num_members)
    max_distance_sq = max_distance * max_distance
    for idx in range(num_members):
        bucket_x, bucket_y = buckets[idx]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_bucket = (bucket_x + dx, bucket_y + dy)
                for other_idx in bucket_map.get(neighbor_bucket, []):
                    if other_idx <= idx:
                        continue
                    dist_x = float(pos_x[idx] - pos_x[other_idx])
                    dist_y = float(pos_y[idx] - pos_y[other_idx])
                    if dist_x * dist_x + dist_y * dist_y <= max_distance_sq:
                        uf.union(idx, other_idx)

    components: Dict[int, List[int]] = {}
    for idx in range(num_members):
        components.setdefault(uf.find(idx), []).append(idx)
    return [
        np.asarray(indices, dtype=np.int64)
        for indices in components.values()
    ]


def spatial_postprocess_clusters(
    cluster_ids,
    pos_x,
    pos_y,
    row_height,
    max_distance_factor=4.0,
) -> torch.Tensor:
    source_device = cluster_ids.device if isinstance(cluster_ids, torch.Tensor) else None
    cluster_ids_np = _to_numpy_int(cluster_ids)
    pos_x_np = _to_numpy_float(pos_x)
    pos_y_np = _to_numpy_float(pos_y)

    if cluster_ids_np.size == 0:
        output = torch.empty(0, dtype=torch.int64)
        return output.to(source_device) if source_device is not None else output

    max_distance = max(float(row_height) * float(max_distance_factor), 1e-12)
    split_cluster_ids = np.empty_like(cluster_ids_np, dtype=np.int64)
    next_cluster_id = 0
    num_original_clusters = 0

    for original_cluster_id in np.unique(cluster_ids_np):
        members = np.flatnonzero(cluster_ids_np == original_cluster_id)
        if members.size == 0:
            continue
        num_original_clusters += 1
        if members.size == 1:
            split_cluster_ids[members] = next_cluster_id
            next_cluster_id += 1
            continue

        components = _split_members_by_distance(
            pos_x_np[members],
            pos_y_np[members],
            max_distance=max_distance,
        )
        for component in components:
            split_cluster_ids[members[component]] = next_cluster_id
            next_cluster_id += 1

    output = torch.from_numpy(split_cluster_ids)
    logger.info(
        "Spatial split modularity clusters: original_clusters=%d split_clusters=%d max_distance=%.3f",
        num_original_clusters,
        next_cluster_id,
        max_distance,
    )
    return output.to(source_device) if source_device is not None else output


def build_topology_leiden_clusters(placedb, params) -> LeidenClusteringResult:
    resolutions = list(getattr(params, "leiden_resolutions", [0.5, 1.0, 2.0]))
    if not resolutions:
        resolutions = [1.0]

    graph = build_star_model_graph(
        flat_net2pin_map=placedb.flat_net2pin_map,
        flat_net2pin_start_map=placedb.flat_net2pin_start_map,
        pin2node_map=placedb.pin2node_map,
        num_movable_nodes=placedb.num_movable_nodes,
        num_nets=placedb.num_nets,
        net_weights=placedb.net_weights,
        ignore_net_degree=getattr(params, "leiden_ignore_net_degree", 100),
    )
    cluster_ids_by_level = run_leiden_multi_resolution(
        graph=graph,
        resolutions=resolutions,
        num_movable_nodes=placedb.num_movable_nodes,
        seed=int(getattr(params, "random_seed", 42)),
    )
    num_clusters_by_level = [
        int(cluster_ids.max().item()) + 1 if cluster_ids.numel() else 0
        for cluster_ids in cluster_ids_by_level
    ]
    return LeidenClusteringResult(
        cluster_ids_by_level=[cluster_ids.cpu() for cluster_ids in cluster_ids_by_level],
        num_clusters_by_level=num_clusters_by_level,
        resolutions_used=[float(resolution) for resolution in resolutions],
    )


def build_active_leiden_clusters(placedb, params, pos) -> LeidenClusteringResult:
    topology_result = getattr(placedb, "modularity_topology_clustering_result", None)
    if topology_result is None:
        topology_result = build_topology_leiden_clusters(placedb, params)

    if not isinstance(pos, torch.Tensor):
        pos = torch.as_tensor(pos)
    num_nodes = int(placedb.num_nodes)
    num_movable_nodes = int(placedb.num_movable_nodes)
    pos_x = pos[:num_movable_nodes]
    pos_y = pos[num_nodes:num_nodes + num_movable_nodes]

    active_cluster_ids_by_level: List[torch.Tensor] = []
    active_num_clusters_by_level: List[int] = []
    split_factor = float(getattr(params, "leiden_spatial_split_factor", 4.0))
    for level_idx, topology_cluster_ids in enumerate(topology_result.cluster_ids_by_level):
        active_cluster_ids = spatial_postprocess_clusters(
            topology_cluster_ids,
            pos_x=pos_x,
            pos_y=pos_y,
            row_height=float(placedb.row_height),
            max_distance_factor=split_factor,
        ).cpu()
        active_cluster_ids_by_level.append(active_cluster_ids)
        num_clusters = int(active_cluster_ids.max().item()) + 1 if active_cluster_ids.numel() else 0
        active_num_clusters_by_level.append(num_clusters)
        logger.info(
            "Active modularity clusters level %d: %s",
            level_idx,
            _cluster_size_stats(active_cluster_ids),
        )

    return LeidenClusteringResult(
        cluster_ids_by_level=active_cluster_ids_by_level,
        num_clusters_by_level=active_num_clusters_by_level,
        resolutions_used=list(topology_result.resolutions_used),
    )


def _resolve_modularity_plot_dir(params) -> str:
    result_dir = getattr(params, "result_dir", "") or os.getcwd()
    design_name = ""
    if hasattr(params, "design_name"):
        try:
            design_name = params.design_name() or ""
        except Exception:
            design_name = ""
    if design_name:
        result_dir = os.path.join(result_dir, design_name)
    plot_dir = os.path.join(result_dir, "plot", "modularity_clusters")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def plot_modularity_clusters(
    placedb,
    params,
    pos,
    clustering_result: LeidenClusteringResult,
    source="active",
    round_idx=0,
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if not isinstance(pos, torch.Tensor):
        pos = torch.as_tensor(pos)

    num_nodes = int(placedb.num_nodes)
    num_movable_nodes = int(placedb.num_movable_nodes)
    pos_x = _to_numpy_float(pos[:num_movable_nodes])
    pos_y = _to_numpy_float(pos[num_nodes:num_nodes + num_movable_nodes])
    node_size_x = _to_numpy_float(placedb.node_size_x[:num_movable_nodes])
    node_size_y = _to_numpy_float(placedb.node_size_y[:num_movable_nodes])
    center_x = pos_x + 0.5 * node_size_x
    center_y = pos_y + 0.5 * node_size_y

    plot_dir = _resolve_modularity_plot_dir(params)
    saved_paths = []
    for level_idx, cluster_ids in enumerate(clustering_result.cluster_ids_by_level):
        cluster_ids_np = _to_numpy_int(cluster_ids)
        if cluster_ids_np.size != num_movable_nodes:
            raise ValueError(
                "modularity cluster plot expects %d movable nodes, got %d"
                % (num_movable_nodes, cluster_ids_np.size)
            )
        if level_idx < len(clustering_result.num_clusters_by_level):
            num_clusters = int(clustering_result.num_clusters_by_level[level_idx])
        else:
            num_clusters = int(cluster_ids_np.max()) + 1 if cluster_ids_np.size else 0
        resolution = (
            clustering_result.resolutions_used[level_idx]
            if level_idx < len(clustering_result.resolutions_used)
            else float("nan")
        )

        output_path = os.path.join(
            plot_dir,
            "modularity_%s_round%d_level%d.png" % (
                source,
                int(round_idx),
                level_idx,
            ),
        )
        npz_path = output_path[:-4] + ".npz"

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(
            center_x,
            center_y,
            c=cluster_ids_np,
            s=2.0,
            cmap="nipy_spectral",
            marker="s",
            linewidths=0,
            alpha=0.9,
            rasterized=True,
        )
        ax.add_patch(
            patches.Rectangle(
                (float(placedb.xl), float(placedb.yl)),
                float(placedb.xh - placedb.xl),
                float(placedb.yh - placedb.yl),
                fill=False,
                edgecolor="black",
                linewidth=1.2,
            )
        )
        ax.set_xlim(float(placedb.xl), float(placedb.xh))
        ax.set_ylim(float(placedb.yl), float(placedb.yh))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(
            "Modularity Clusters (%s) L%d | clusters=%d | resolution=%s"
            % (
                source,
                level_idx,
                num_clusters,
                ("%.4f" % resolution) if math.isfinite(resolution) else "n/a",
            )
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        np.savez_compressed(
            npz_path,
            cluster_ids=cluster_ids_np,
            center_x=center_x,
            center_y=center_y,
            num_clusters=np.asarray([num_clusters], dtype=np.int64),
            resolution=np.asarray([resolution], dtype=np.float64),
        )
        saved_paths.append(output_path)
        logger.info(
            "Saved modularity cluster debug plot: source=%s round=%d level=%d clusters=%d path=%s",
            source,
            int(round_idx),
            level_idx,
            num_clusters,
            output_path,
        )

    return saved_paths


__all__ = [
    "LeidenClusteringResult",
    "build_star_model_graph",
    "run_leiden_multi_resolution",
    "spatial_postprocess_clusters",
    "build_topology_leiden_clusters",
    "build_active_leiden_clusters",
    "plot_modularity_clusters",
    "normalize_cluster_ids",
]

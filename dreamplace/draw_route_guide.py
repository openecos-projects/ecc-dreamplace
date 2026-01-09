import matplotlib.pyplot as plt
import os
from matplotlib.collections import LineCollection

def draw_route_guide(guide_path, output_path):
    wires = []
    current_net = None
    
    print(f"Reading guide from: {guide_path}")
    if not os.path.exists(guide_path):
        print(f"Error: File not found: {guide_path}")
        return

    with open(guide_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            if parts[0] == 'guide':
                current_net = parts[1]
            elif parts[0] == 'wire':
                # Format: wire grid1_x grid1_y grid2_x grid2_y real1_x real1_y real2_x real2_y layer
                try:
                    x1 = float(parts[5])
                    y1 = float(parts[6])
                    x2 = float(parts[7])
                    y2 = float(parts[8])
                    layer = parts[9]
                    wires.append({'coords': [(x1, y1), (x2, y2)], 'layer': layer, 'net': current_net})
                except (ValueError, IndexError):
                    continue
    
    print(f"Found {len(wires)} wire segments.")
    
    if not wires:
        print("No wires found to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Group by layer for plotting efficiency
    layer_groups = {}
    for w in wires:
        l = w['layer']
        if l not in layer_groups:
            layer_groups[l] = []
        layer_groups[l].append(w['coords'])
        
    # Define some colors
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    layer_names = sorted(layer_groups.keys())
    
    print("Plotting layers...")
    for i, layer in enumerate(layer_names):
        segments = layer_groups[layer]
        color = colors[i % len(colors)]
        lc = LineCollection(segments, colors=color, linewidths=0.5, label=layer, alpha=0.6)
        ax.add_collection(lc)
    
    ax.autoscale()
    ax.set_aspect('equal')
    plt.legend(loc='upper right')
    plt.title(f"Route Guide Visualization: {os.path.basename(guide_path)}")
    plt.xlabel("X (microns)")
    plt.ylabel("Y (microns)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    print(f"Saving plot to {output_path}")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print("Done.")

if __name__ == "__main__":
    guide_file = "/nfs/share/home/sxr/routability_benchmark/dataset_cx55/20251023/gcd/workspace/output/iEDA/data/rt/rt_temp_directory/early_router/route.guide"
    output_file = "route_guide_plot.png"
    draw_route_guide(guide_file, output_file)

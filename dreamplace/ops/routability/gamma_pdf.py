import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln # Using gammaln for numerical stability

def shortest_path_pdf(x, y, W, H):
    """
    Calculates the probability density for a shortest right-angled path
    from (0,0) to (W,H) to pass through a point (x,y).
    Note: This returns an unnormalized probability density value,
          which is sufficient for visualizing the shape of the distribution.
    """
    # Use a small epsilon to avoid potential issues at the boundaries W and H.
    # While gammaln can handle 0, this adds robustness.
    epsilon = 1e-10
    x = np.clip(x, epsilon, W - epsilon)
    y = np.clip(y, epsilon, H - epsilon)

    # Calculate the formula in its logarithmic form for stability.
    # log(p(x,y)) ∝ log(C(x+y, x)) + log(C(W-x+H-y, W-x))
    # where log(C(n,k)) = gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)
    
    # Part 1: From (0,0) to (x,y)
    log_p1 = gammaln(x + y + 1) - gammaln(x + 1) - gammaln(y + 1)
    
    # Part 2: From (x,y) to (W,H)
    log_p2 = gammaln(W - x + H - y + 1) - gammaln(W - x + 1) - gammaln(H - y + 1)
    
    # Add the log-probabilities together and then take the exponent to get the final value.
    # We ignore the normalization constant C as it doesn't affect the shape.
    return np.exp(log_p1 + log_p2)

# --- 1. Parameter Setup ---
W, H = 1.0, 1.0  # Width and Height of the rectangle (a unit square)
grid_points = 200 # Density of the grid; higher values result in a finer image

# --- 2. Grid Creation ---
x_vals = np.linspace(0, W, grid_points)
y_vals = np.linspace(0, H, grid_points)
X, Y = np.meshgrid(x_vals, y_vals)

# --- 3. Calculate the PDF for each point on the grid ---
Z = shortest_path_pdf(X, Y, W, H)

# --- 4. Visualization ---

# This line ensures that the minus sign is displayed correctly
plt.rcParams['axes.unicode_minus'] = False

# **Visualization 1: 2D Heatmap with Contour Lines**
fig1, ax1 = plt.subplots(figsize=(8, 7))
# Use pcolormesh to draw the heatmap
im = ax1.pcolormesh(X, Y, Z, shading='auto', cmap='plasma')
# Add a color bar
cbar = fig1.colorbar(im, ax=ax1)
cbar.set_label('Relative Probability Density', rotation=270, labelpad=15)
# Add contour lines
contour = ax1.contour(X, Y, Z, levels=10, colors='white', alpha=0.5)
ax1.clabel(contour, inline=True, fontsize=8)

ax1.set_xlabel('X Coordinate')
ax1.set_ylabel('Y Coordinate')
ax1.set_title(f'Shortest Path Probability Density (W={W}, H={H})\n2D Heatmap')
ax1.set_aspect('equal') # Ensure x and y axes have the same scale
plt.savefig("shortest_path_density_2D.png", dpi=300)
plt.show()


# **Visualization 2: 3D Surface Plot**
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')

# Draw the 3D surface
surf = ax2.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none')

ax2.set_xlabel('X Coordinate')
ax2.set_ylabel('Y Coordinate')
ax2.set_zlabel('Relative Probability Density')
ax2.set_title(f'Shortest Path Probability Density (W={W}, H={H})\n3D "Mountain Ridge" Visualization')
# Add a color bar
cbar2 = fig2.colorbar(surf, shrink=0.5, aspect=5)
cbar2.set_label('Relative Probability Density')
plt.savefig("shortest_path_density_3D.png", dpi=300)
plt.show()
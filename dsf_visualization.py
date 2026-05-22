import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================
# Load CSV
# ============================================
csv_path = "/Users/dnbn/code/DSF_TriangleMesh/dsf_vertices_all.csv"

df = pd.read_csv(csv_path)

# ============================================
# Unique shape types
# ============================================
shape_types = df["shape_type"].unique()

# ============================================
# Visualization
# ============================================
fig = plt.figure(figsize=(12, 10))

for plot_idx, shape in enumerate(shape_types, start=1):

    subdf = df[df["shape_type"] == shape]

    ax = fig.add_subplot(2, 2, plot_idx, projection="3d")

    x = subdf["v_x"].values
    y = subdf["v_y"].values
    z = subdf["v_z"].values

    ax.scatter(x, y, z, s=40)

    # index label
    for _, row in subdf.iterrows():
        ax.text(
            row["v_x"],
            row["v_y"],
            row["v_z"],
            str(int(row["index"])),
            fontsize=7
        )

    R = subdf["R"].iloc[0]

    ax.set_title(f"{shape}, R={R:.3g}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    max_range = max(
        abs(x).max(),
        abs(y).max(),
        abs(z).max()
    )

    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

plt.tight_layout()

# save figure
output_image = "/Users/dnbn/code/DSF_TriangleMesh/dsf_vertices_visualization.png"

plt.savefig(output_image, dpi=300)

print(f"Saved: {output_image}")

plt.show()
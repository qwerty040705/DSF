import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# Load CSV
# ============================================
csv_path = "/Users/dnbn/code/DSF_TriangleMesh/dsf_vertices_all.csv"

df = pd.read_csv(csv_path)

# ============================================
# Shape order
# ============================================
shape_order = [
    "round",
    "normal",
    "long",
    "very_long",
    "extreme_long",
]

shape_types = [s for s in shape_order if s in df["shape_type"].unique()]

# ============================================
# Visualization
# ============================================
fig = plt.figure(figsize=(18, 10))

for plot_idx, shape in enumerate(shape_types, start=1):

    subdf = df[df["shape_type"] == shape]

    ax = fig.add_subplot(2, 3, plot_idx, projection="3d")

    x = subdf["v_x"].values
    y = subdf["v_y"].values
    z = subdf["v_z"].values

    ax.scatter(x, y, z, s=40)

    # Draw vectors from origin
    for _, row in subdf.iterrows():
        ax.plot(
            [0, row["v_x"]],
            [0, row["v_y"]],
            [0, row["v_z"]],
            linewidth=0.6,
            alpha=0.5
        )

        ax.text(
            row["v_x"],
            row["v_y"],
            row["v_z"],
            str(int(row["index"])),
            fontsize=7
        )

    R = subdf["R"].iloc[0]
    log10_R = np.log10(R)

    ax.set_title(f"{shape}\nR={R:.4g}, log10(R)={log10_R:.1f}")

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

# Empty 6th subplot
ax_empty = fig.add_subplot(2, 3, 6)
ax_empty.axis("off")

plt.tight_layout()

# save figure
output_image = "/Users/dnbn/code/DSF_TriangleMesh/dsf_vertices_visualization.png"

plt.savefig(output_image, dpi=300)

print(f"Saved: {output_image}")

plt.show()
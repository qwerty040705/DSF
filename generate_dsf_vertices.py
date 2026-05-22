import numpy as np
import pandas as pd

# =========================
# Settings
# =========================
N = 32

R_list = [
    ("round", 1.0),
    ("normal", np.sqrt(10)),
    ("long", 10.0),
    ("very_long", 10.0 * np.sqrt(10)),
    ("extreme_long", 100.0),
]

output_file = "dsf_vertices_all.csv"

# =========================
# Fixed direction u_i
# =========================
U = np.zeros((N, 3))

# v_min direction
U[0] = np.array([0.0, 0.0, -1.0])

golden_angle = np.pi * (3.0 - np.sqrt(5.0))

# u_2 ~ u_32 : upper hemisphere
for idx in range(1, N):
    k = idx - 1

    z = (k + 0.5) / (N - 1)
    rho = np.sqrt(1.0 - z**2)
    theta = k * golden_angle

    U[idx] = np.array([
        rho * np.cos(theta),
        rho * np.sin(theta),
        z
    ])

# =========================
# Generate all vertices
# =========================
all_rows = []

for label, R in R_list:

    # symmetric around 1
    # r_1  = 1 / sqrt(R)
    # r_32 = sqrt(R)
    # r_32 / r_1 = R
    exponent = np.arange(N) / (N - 1) - 0.5
    r = R ** exponent

    V = U * r[:, None]

    for i in range(N):
        all_rows.append({
            "shape_type": label,
            "R": R,
            "log10_R": np.log10(R),
            "index": i + 1,

            "r_i": r[i],

            "u_x": U[i, 0],
            "u_y": U[i, 1],
            "u_z": U[i, 2],

            "v_x": V[i, 0],
            "v_y": V[i, 1],
            "v_z": V[i, 2],

            "norm_v": np.linalg.norm(V[i]),
        })

# =========================
# Save CSV
# =========================
df = pd.DataFrame(all_rows)
df.to_csv(output_file, index=False)

print(f"Saved: {output_file}")
print(df.groupby("shape_type")["norm_v"].agg(["min", "max"]))
from pathlib import Path

import numpy as np


# =========================================================
# Generate DSF vertices only.
#
# Each output CSV stores exactly 32 vertices with x,y,z columns.
# Scale labels are encoded in the file names, not as extra CSV columns.
# =========================================================

RNG_SEED = 1
NUM_VERTICES = 32
DIM = 3

SCALE_CASES = [
    ("verysmall", -3, -1),
    ("small", -2, 0),
    ("normal", -1, 1),
    ("large", 0, 2),
    ("verylarge", 1, 3),
]

OUTPUT_DIR = Path("dsf_vertices_xyz")


def random_unit_vectors(rng: np.random.Generator, count: int) -> np.ndarray:
    vectors = rng.normal(size=(count, DIM))
    norms = np.linalg.norm(vectors, axis=1)

    bad = norms < 1e-12
    while np.any(bad):
        vectors[bad] = rng.normal(size=(np.count_nonzero(bad), DIM))
        norms = np.linalg.norm(vectors, axis=1)
        bad = norms < 1e-12

    return vectors / norms[:, None]


def generate_base_vertices(
    rng: np.random.Generator,
    log_radius_min: float,
    log_radius_max: float,
) -> np.ndarray:
    directions = random_unit_vectors(rng, NUM_VERTICES)
    log_radii = rng.uniform(log_radius_min, log_radius_max, size=NUM_VERTICES)
    radii = np.power(10.0, log_radii)
    return directions * radii[:, None]


def save_vertices(path: Path, vertices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        vertices,
        delimiter=",",
        header="x,y,z",
        comments="",
        fmt="%.17g",
    )


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    saved_count = 0

    for scale_label, log_radius_min, log_radius_max in SCALE_CASES:
        base_vertices = generate_base_vertices(rng, log_radius_min, log_radius_max)
        save_vertices(OUTPUT_DIR / f"{scale_label}.csv", base_vertices)
        saved_count += 1

    print(f"Saved {saved_count} vertex files under: {OUTPUT_DIR}")
    print(
        f"Each file contains {NUM_VERTICES} rows and only x,y,z coordinate columns."
    )


if __name__ == "__main__":
    main()

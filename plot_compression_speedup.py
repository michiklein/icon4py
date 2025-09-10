import matplotlib.pyplot as plt

# Data (compression %, speedup)
data_main = {
    "c2e2c": (50, 1.61),
    "c2e2v": (50, 1.69),
    "c2v2c": (33, 2.08),
    "c2v2e": (17, 1.84),
    "e2c2v": (33, 1.58),
    "e2v2c": (17, 1.57),
    "e2v2e": (17, 1.49),
    "v2c2e": (33, 1.09),
    "v2c2v": (67, 3.06),
    "v2e2c": (50, 1.23),
    "v2e2v": (50, 2.32),
}

data_long = {
    "c2e2c2e2c": (33, 1.87),
    "v2e2c2v": (75, 3.81),
}

plt.figure(figsize=(7, 5))

# Main kernels
for name, (x, y) in data_main.items():
    plt.scatter(x, y, color="C0")
    plt.annotate(name, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)

# Long kernels
for name, (x, y) in data_long.items():
    plt.scatter(x, y, color="C1", marker="x")
    plt.annotate(name, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)

plt.xlabel("Compression (%)")
plt.ylabel("Speedup (off/on)")
plt.title("Compression vs. Speedup")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("compression_vs_speedup.png", dpi=200, bbox_inches="tight")
plt.show()

import os
import re
import matplotlib.pyplot as plt


def load_timings(appendix):
    filename = f"results/neighbor_sums_{appendix}.txt"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, "r") as f:
        lines = f.readlines()

    timings = {}
    parsing = False
    for line in lines:
        if line.strip().startswith("Summary of neighbor combinations"):
            parsing = True
            continue
        if parsing and line.strip() == "":
            break
        if parsing:
            match = re.match(r"(\w+)->(\w+): ([\d.]+)s", line.strip())
            if match:
                combo = f"{match.group(1)}->{match.group(2)}"
                time_val = float(match.group(3))
                timings[combo] = time_val

    return timings
 

def plot_comparison(timings1, timings2, label1, label2):
    keys = sorted(set(timings1.keys()) | set(timings2.keys()))
    values1 = [timings1.get(k, 0.0) for k in keys]
    values2 = [timings2.get(k, 0.0) for k in keys]

    x = range(len(keys))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], values1, width, label=label1)
    plt.bar([i + width / 2 for i in x], values2, width, label=label2)

    plt.xlabel("Neighbor Combination")
    plt.ylabel("Execution Time (s)")
    plt.title("Comparison of Neighbor Sum Timings")
    plt.xticks(ticks=x, labels=keys, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    os.makedirs("results", exist_ok=True)
    out_path = f"results/plot_{label1}_{label2}.png"
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

    plt.show()


def main():
    a1 = input("Enter first results appendix (e.g., test1): ").strip()
    a2 = input("Enter second results appendix (e.g., test2): ").strip()

    try:
        t1 = load_timings(a1)
        t2 = load_timings(a2)
    except FileNotFoundError as e:
        print(e)
        return

    plot_comparison(t1, t2, a1, a2)


if __name__ == "__main__":
    main()

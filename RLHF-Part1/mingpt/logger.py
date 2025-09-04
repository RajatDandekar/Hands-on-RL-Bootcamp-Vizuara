from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt


class Logger:
    def __init__(self):
        self.logs = defaultdict(lambda: {"idx": [], "values": []})

    def log(self, key, iter_num, value):
        self.logs[key]["idx"].append(iter_num)
        self.logs[key]["values"].append(value)

    def plot(self, key_groups: Dict[str, List[str]], filename=None):
        fig, axes = plt.subplots(1, len(key_groups), figsize=(6 * len(key_groups), 5))

        if len(key_groups) == 1:
            axes = [axes]

        for ax, (title, keys) in zip(axes, key_groups.items()):
            for key in keys:
                data = self.logs[key]
                ax.plot(data["idx"], data["values"], label=key)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(title)
            ax.set_title(title)
            if len(keys) > 1:
                ax.legend()

        fig.tight_layout()
        if filename:
            fig.savefig(filename)

        return fig

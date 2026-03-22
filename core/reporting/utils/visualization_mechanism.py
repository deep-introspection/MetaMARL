import matplotlib.pyplot as plt


def plot_es_metrics(es_metrics_history, save_path=None):
    fig, ax = plt.subplots()
    # ... draw lines ...
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig

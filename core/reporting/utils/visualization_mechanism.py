import matplotlib.pyplot as plt


def plot_es_metrics(es_metrics_history, save_path=None):
    """Plot Evolution Strategies metrics over generations.

    Stub implementation — currently creates an empty axes object.  Extend with
    actual metric series (e.g. best fitness, collapse rate) as needed.

    Parameters
    ----------
    es_metrics_history : list[dict[str, Any]]
        One dict per ES generation containing metric values to plot.
    save_path : str or None
        If provided, the figure is saved to this path at 200 dpi.

    Returns
    -------
    matplotlib.figure.Figure
        The (currently empty) figure object.
    """
    fig, ax = plt.subplots()
    # ... draw lines ...
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig

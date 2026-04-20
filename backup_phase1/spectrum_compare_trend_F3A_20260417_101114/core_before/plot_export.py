import plotly.io as pio


def export_plot_png(fig):
    return pio.to_image(
        fig,
        format="png",
        width=1920,
        height=1080,
        scale=2
    )
"""
Author: Benedikt Goodman, Division for National Accounts, Statistics Norway
Email: bgo@ssb.no
Created: 22/04/2024
"""

import seaborn as sns
import matplotlib.pyplot as plt


def plot_multiple_lines(
    df, x, y_columns, title="Line Plot", xaxis_title="X Axis", yaxis_title="Y Axis"
):
    """
    Plot multiple lines from the same DataFrame on a single figure using Seaborn, with specified columns for the x and multiple y axes.

    Parameters:
    - df : pandas.DataFrame
        The DataFrame containing the data to plot.
    - x : str
        The column name for the x-axis data.
    - y_columns : list[str]
        A list of column names for the y-axis data for each line plot.
    - title : str, optional
        Title of the plot.
    - xaxis_title : str, optional
        Title for the X-axis.
    - yaxis_title : str, optional
        Title for the Y-axis.

    Returns:
    - None, but displays a matplotlib figure.
    """
    # Set the color palette to viridis and the theme
    sns.set_theme(style="darkgrid")
    sns.set_context("paper", rc={"axes.titlesize": 16})
    palette = sns.color_palette("viridis", n_colors=len(y_columns))

    # Create the figure and the axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting lines for each specified y column
    for i, y in enumerate(y_columns):
        sns.lineplot(
            data=df, x=x, y=y, color=palette[i], label=f"{y}", ax=ax, errorbar=None
        )

    # Setting the plot title and labels
    ax.set_title(title)
    ax.set_xlabel(xaxis_title)
    ax.set_ylabel(yaxis_title)

    # Show legend
    ax.legend()

    # Return the figure object
    return fig

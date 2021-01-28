"""
Author: Ryan Friedman (@rfriedman22)
Email: ryan.friedman@wustl.edu
"""

from datetime import datetime
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import auc


def set_presentation_params():
    """Set the matplotlib rcParams to values for presentation-size figures. (Misnomer because I don't use this.)
    
    """
    mpl.rcParams["axes.titlesize"] = 90
    mpl.rcParams["axes.labelsize"] = 80
    mpl.rcParams["xtick.labelsize"] = 60
    mpl.rcParams["ytick.labelsize"] = 60
    mpl.rcParams["legend.fontsize"] = 60
    mpl.rcParams["figure.figsize"] = (25, 25)
    mpl.rcParams["image.cmap"] = "viridis"
    mpl.rcParams["lines.markersize"] = 14
    mpl.rcParams["lines.linewidth"] = 15
    mpl.rcParams["font.size"] = 60
    mpl.rcParams["xtick.major.size"] = 10
    mpl.rcParams["xtick.major.width"] = 3
    mpl.rcParams["ytick.major.size"] = 10
    mpl.rcParams["ytick.major.width"] = 3
    
    
def set_print_params():
    """Set the matplotlib rcParams to values for print-size figures. (Misnomer because I use this in slides.)
    
    """
    mpl.rcParams["axes.titlesize"] = 25
    mpl.rcParams["axes.labelsize"] = 20
    mpl.rcParams["xtick.labelsize"] = 15
    mpl.rcParams["ytick.labelsize"] = 15
    mpl.rcParams["legend.fontsize"] = 15
    mpl.rcParams["figure.figsize"] = (8, 8)
    mpl.rcParams["image.cmap"] = "viridis"
    mpl.rcParams["lines.markersize"] = 3
    mpl.rcParams["lines.linewidth"] = 3
    mpl.rcParams["font.size"] = 15


def set_manuscript_params():
    """Set the matplotlib rcParams to values for manuscript-size figures.

    """
    mpl.rcParams["figure.figsize"] = (4, 4)
    mpl.rcParams["axes.titlesize"] = 15
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["legend.fontsize"] = 12
    mpl.rcParams["image.cmap"] = "viridis"
    mpl.rcParams["lines.markersize"] = 1.25
    mpl.rcParams["lines.linewidth"] = 2
    mpl.rcParams["font.size"] = 12
    mpl.rcParams["savefig.dpi"] = 200


def add_letter(ax, x, y, letter):
    """Add a letter to label an axes as a panel of a larger figure.

    Parameters
    ----------
    ax : Axes object
        The panel to add the letter to.
    x : int
        x coordinate of the right side of the letter, in ax.transAxes coordinates
    y : int
        y coordinate of the top side of the letter, in ax.transAxes coordinates
    letter : str
        The letter to add

    Returns
    -------
    Text
        The created Text instance
    """
    return ax.text(x, y, letter, fontsize=mpl.rcParams["axes.labelsize"], fontweight="bold", ha="right", va="top",
                   transform=ax.transAxes)


def rotate_ticks(ticks, rotation=90):
    """Rotate tick labels from an Axes object after the ticks were already generated.

    Parameters
    ----------
    ticks : list[Text]
        The tick labels to rotate
    rotation : int or float
        The angle to set for the tick labels

    Returns
    -------
    None
    """
    for tick in ticks:
        tick.set_rotation(rotation)


def set_color(values):
    """A wrapper for converting numbers into colors. Given a number between 0 and 1, convert it to the corresponding color in the color scheme.
    
    """
    my_cmap = mpl.cm.get_cmap()
    return my_cmap(values)


def save_fig(fig, prefix, tight_layout=True, timestamp=True, tight_pad=1.08):
    """Save a figure as a PNG and an SVG.
    
    """
    if tight_layout:
        fig.tight_layout(pad=tight_pad)
    if timestamp:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0, 0, now, transform=fig.transFigure)
    fig.savefig(f"{prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{prefix}.png", bbox_inches="tight")
    
    
def setup_multiplot(n_plots, n_cols=2, sharex=True, sharey=True, big_dimensions=True):
    """Setup a multiplot and hide any superfluous axes that may result.

    Parameters
    ----------
    n_plots : int
        Number of subplots to make
    n_cols : int
        Number of columns in the multiplot. Number of rows is inferred.
    sharex : bool
        Indicate if the x-axis should be shared.
    sharey : bool
        Indicate if the y-axis should be shared.
    big_dimensions : bool
        If True, then the size of the multiplot is the default figure size multiplied by the number of rows/columns.
        If False, then the entire figure is the default figure size.

    Returns
    -------
    fig : figure handle
    ax_list : list-like
        The list returned by plt.subplots(), but any superfluous axes are removed and replaced by None
    """
    n_rows = int(np.ceil(n_plots / n_cols))
    row_size, col_size = mpl.rcParams["figure.figsize"]

    if big_dimensions:
        # A bit counter-intuitive...the SIZE of the row is the width, which depends on the number of columns
        row_size *= n_cols
        col_size *= n_rows

    fig, ax_list = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(row_size, col_size), sharex=sharex, sharey=sharey)

    # The index corresponding to n_plots is the first subplot to be hidden
    for i in range(ax_list.size):
        coords = np.unravel_index(i, ax_list.shape)
        ax = ax_list[coords]
        if i >= n_plots:
            ax.remove()
            ax_list[coords] = None

    return fig, ax_list


def volcano_plot(df, x_col, y_col, colors, alpha=1, xaxis_label=None, yaxis_label=None, title=None, figname=None,
                 xline=None, yline=None, xticks=None, vmin=None, vmax=None, cmap=None, colorbar=False, figax=None):
    """Make a volcano plot, without transforming the x-axis but taking -log10 of the y-axis. Assign different points
    different colors to highlight different classes.

    Parameters
    ----------
    df : pd.DataFrame
    x_col : str
        Column of the df to plot on x
    y_col : str
        Column of the df to plot on y. Take -log10 of this column before plotting
    colors : list-like
        Indicates color to use for each row of df.
    alpha : float
        Opacity of the points.
    xaxis_label : str
        If specified, the label for the x-axis. Otherwise use x_col.
    yaxis_label : str
        If specified, the label for the y-axis. Otherwise use y_col.
    title : str
        If specified, make a title for the plot.
    figname : str
        If specified, save the figure with this name.
    xline : int or float or list
        If specified, plot a dashed vertical line at x = xline
    yline : int or float or list
        If specified, plot a dashed horizontal line at y = yline
    xticks : list
        If specified, set the x ticks to these values.
    vmin : int or float
        If specified, minimum value for the colormap.
    vmax : int or float
        If specified, maximum value for the colormap.
    cmap : str
        If specified, use this colormap. Otherwise, use the default.
    colorbar : bool
        If True, display a colorbar to the right.
    figax : (figure, axes) or None
        If specified, make the plot in the provided axes. Otherwise, generate a new axes.

    Returns
    -------
    fig : Figure handle
    """
    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    # Prepare the data
    x = df[x_col]
    y = -np.log10(df[y_col])
    scatter_kwargs = {"c": colors, "alpha": alpha}
    if vmin:
        scatter_kwargs["vmin"] = vmin
    if vmax:
        scatter_kwargs["vmax"] = vmax
    if cmap:
        scatter_kwargs["cmap"] = cmap

    scatterplot = ax.scatter(x, y, **scatter_kwargs)

    # Default axis labels if none specified
    if not xaxis_label:
        xaxis_label = x_col
    if not yaxis_label:
        yaxis_label = f"-log10 {y_col}"

    # Add dotted lines if specified
    line_kwargs = {"linestyle": "--", "color": "black"}
    if xline is not None:
        if type(xline) is list:
            for xl in xline:
                ax.axhline(xl, **line_kwargs)
        else:
            ax.axhline(xline, **line_kwargs)
    if yline is not None:
        if type(yline) is list:
            for yl in yline:
                ax.axvline(yl, **line_kwargs)
        else:
            ax.axvline(yline, **line_kwargs)

    # Axis labels, ticks, colorbar, title if specified
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)

    if xticks is not None:
        ax.set_xticks(xticks)

    if colorbar:
        fig.colorbar(scatterplot, orientation="vertical")

    if title:
        ax.set_title(title)

    if figname:
        save_fig(fig, figname)

    return fig


def scatter_with_corr(x, y, xlabel, ylabel, colors="black", xticks=None, yticks=None, loc=None, figname=None,
                      alpha=1.0, figax=None):
    """Make a scatter plot and display the correlation coefficients in a specified location.

    Parameters
    ----------
    x : list-like
        Data to plot on the x axis.
    y : list-like
        Data to plot on the y axis.
    xlabel : str
        Label for the x axis.
    ylabel : str
        Label for the y axis.
    colors : "density", str or list-like
        If "density", color points based on point density in 2D space. If another str, make every point the same
        color. If list-like, specifies the color for each point.
    xticks : list-like
        If specified, set the x axis ticks to these values.
    yticks: list-like
        If specified, set the y axis ticks to these values.
    loc : str, must be one of "upper left", "upper right", "lower left", or "lower right"
        The location of the plot to display the correlations. If None, just print to the screen. If some other
        string, assume "lower right".
    figname : str
        If specified, save the figure with this name.
    alpha : float
        Alpha (opacity) of the points.
    figax : (figure, axes) or None
        If specified, make the plot in the provided axes. Otherwise, generate a new axes.

    Returns
    -------
    fig : Figure handle
    ax : Axes handle
    """
    # Correlations
    pcc, _ = stats.pearsonr(x, y)
    scc, _ = stats.spearmanr(x, y)
    n = len(x)
    text = f"PCC = {pcc:.3f}\nSCC = {scc:.3f}\nn = {n}"

    # Calculate the density to display on the scatter plot, if specified
    if type(colors) is str and colors == "density":
        xy = np.vstack([x, y])
        colors = stats.gaussian_kde(xy)(xy)
        order = colors.argsort()
        x, y, colors = x[order], y[order], colors[order]
        colors = set_color(colors / colors.max())

    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    ax.scatter(x, y, color=colors, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    # If no location is specified for the correlations, print to screen.
    if loc is None:
        print(text)
    # Parse info on location
    else:
        yloc, xloc = loc.split()
        if yloc == "upper":
            yloc = 0.98
            va = "top"
        else:
            if yloc != "lower":
                print("Warning, did not recognize yloc, assuming lower")
            yloc = 0.02
            va = "bottom"

        if xloc == "left":
            xloc = 0.02
            ha = "left"
        else:
            if xloc != "right":
                print("Warning, did not recognize xloc, assuming right")
            xloc = 0.98
            ha = "right"

        ax.text(xloc, yloc, text, ha=ha, va=va, transform=ax.transAxes)

    if figname:
        save_fig(fig, figname)

    return fig, ax


# LEGACY FUNCTION
def violin_plot_series(ser, class_masks, class_names, yname, class_colors=None, alpha=1.0, transformation_function=None,
                       pseudocount=0, figname=None, vert=True, yticks=None, figax=None, **kwargs):
    """ Make a violin plot from a series, with len(class_masks) violins.

    Parameters
    ----------
    ser : pd.Series
        Series containing the data
    class_masks : list of pd.Series
        Each value of the list is a boolean mask corresponding to different subsets of the Series.
    class_names : list[str]
        Names for each class
    yname : str
        Name for the y axis
    class_colors : list
        Optional colors for each group.
    alpha : float
        Opacity of the violins.
    transformation_function : function handle
        Optional transformation to apply to the data.
    pseudocount : int or float
        Optional pseudocount for the data.
    figname : str
        If specified, save the figure to a file with this name.
    vert : bool
        If True, violins are vertical. Otherwise, violins are horizontal.
    yticks : list
        If specified, indicates the ticks for the y axis.
    figax : (figure, axes) or None
        If specified, make the plot in the provided axes. Otherwise, generate a new axes.
    kwargs : dict
        Arguments for saving the figure

    Returns
    -------
    fig : figure handle
    """
    data = [ser[i] for i in class_masks]
    fig = _make_violin_plot(data, class_names, yname, colors=class_colors, alpha=alpha,
                            transformation_function=transformation_function, pseudocount=pseudocount,
                            figname=figname, vert=vert, yticks=yticks, figax=figax, **kwargs)
    return fig


def violin_plot_groupby(grouper, yname, class_names=None, class_colors=None, alpha=1.0, transformation_function=None,
                        pseudocount=0, figname=None, vert=True, yticks=None, figax=None, **kwargs):
    """Make a violin plot from a groupby object.

    Parameters
    ----------
    grouper : pd.DataFrameGroupBy or pd.SeriesGroupBy
        Group by object where each group is data for a different violin.
    yname : str
        Name for the y axis
    class_names : list
        Optional names for each group. If not specified, use the names from the grouper
    class_colors : list
        Optional colors for each group.
    alpha : float
        Opacity of the violins.
    transformation_function : function handle
        Optional transformation to apply to the data.
    pseudocount : int or float
        Optional pseudocount for the data.
    figname : str
        If specified, save the figure to a file with this name.
    vert : bool
        If True, violins are vertical. Otherwise, violins are horizontal.
    yticks : list
        If specified, indicates the ticks for the y axis.
    figax : (figure, axes) or None
        If specified, make the plot in the provided axes. Otherwise, generate a new axes.
    kwargs : dict
        Arguments for saving the figure

    Returns
    -------
    fig : figure handle
    """
    names, data = zip(*[(i, j) for i, j in grouper if len(j) > 0])
    if class_names:
        names = class_names

    fig = _make_violin_plot(data, names, yname, colors=class_colors, alpha=alpha,
                            transformation_function=transformation_function, pseudocount=pseudocount,
                            figname=figname, vert=vert, yticks=yticks, figax=figax, **kwargs)
    return fig


# LEGACY FUNCTION
def violin_plot_by_column(df, y_label, column_colors=None, alpha=1.0, transformation_function=None, pseudocount=0,
                          figname=None, vert=True, xnames=None, yticks=None, figax=None):
    """Make a violin plot for each column of a dataframe"""
    data_values = [df[i] for i in df]
    if xnames is None:
        xnames = df.columns

    fig = _make_violin_plot(data_values, xnames, y_label, colors=column_colors, alpha=alpha,
                            transformation_function=transformation_function, pseudocount=pseudocount, figname=figname,
                            vert=vert, yticks=yticks, figax=figax)
    return fig


# LEGACY FUNCTION
def violin_plot(df, class_masks, class_names, column_name, class_colors=None, alpha=1.0, transformation_function=None,
                pseudocount=0, y_label=None, figname=None, vert=True, yticks=None, figax=None, **kwargs):
    """Make a violin plot with len(class_masks) violins for the specified column from a DataFrame.
    
    """
    data_values = [df.loc[i, column_name].values for i in class_masks]
    if not y_label:
        y_label = column_name
    
    fig = _make_violin_plot(data_values, class_names, y_label, colors=class_colors, alpha=alpha,
                            transformation_function=transformation_function, pseudocount=pseudocount,
                            figname=figname, vert=vert, yticks=yticks, figax=figax, **kwargs)
    return fig


def _make_violin_plot(data_values, x_labels, y_label, colors=None, alpha=1.0, transformation_function=None,
                      pseudocount=0, figname=None, vert=True, yticks=None, whisker=1.5, figax=None, **kwargs):
    """Helper function to make violin plots"""
    # Transform the data (e.g. take the log10) if necessary
    if transformation_function:
        data_values = [transformation_function(i + pseudocount) for i in data_values]
    xaxis = np.arange(len(x_labels)) + 1

    # Set the color to grey for everything if colors aren't specified
    if colors is None:
        colors = ["grey"] * len(x_labels)

    # Separate outliers from the rest
    class_quartiles = np.array([np.percentile(i, [25, 50, 75]) for i in data_values])
    class_iqrs = class_quartiles[:, 2] - class_quartiles[:, 0]
    class_whisker = class_iqrs * whisker
    outlier_masks = [(group_data > quartiles[2] + whisk) | (group_data < quartiles[0] - whisk)
            for group_data, quartiles, whisk in zip(data_values, class_quartiles, class_whisker)]
    outlier_data = [group_data[group_mask] for group_data, group_mask in zip(data_values, outlier_masks)]
    main_data = [group_data[~group_mask] for group_data, group_mask in zip(data_values, outlier_masks)]

    # Plot the data and color the violins accordingly.
    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    parts = ax.violinplot(main_data, vert=vert)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(alpha)

    # Clean up the plot
    parts["cmins"].remove()
    parts["cmaxes"].remove()
    parts["cbars"].remove()

    # Add lines for median
    if vert:
        ax.hlines(class_quartiles[:, 1], xaxis - 0.2, xaxis + 0.2, colors="black", zorder=3,
                  lw=mpl.rcParams["lines.linewidth"] * 2)
        # Old code to show a box for the IQR and a dot for the median
        # ax.scatter(xaxis, class_quartiles[:, 1], marker="o", color="white", zorder=3,
        #            s=mpl.rcParams["lines.markersize"] * 10)
        #ax.vlines(xaxis, class_quartiles[:, 0], class_quartiles[:, 2],
        #          color="black", zorder=1)

        # Plot outliers
        for x, outliers in zip(xaxis, outlier_data):
            ax.scatter([x] * len(outliers), outliers, color="k")

        ax.set_ylabel(y_label)
        ax.set_xticks(xaxis)
        ax.set_xticklabels(x_labels)
        if yticks is not None:
            ax.set_yticks(yticks)
    else:
        ax.vlines(class_quartiles[:, 1], xaxis - 0.2, xaxis + 0.2, colors="black", zorder=3,
                  lw=mpl.rcParams["lines.linewidth"] * 2)
        # ax.scatter(class_quartiles[:, 1], xaxis, marker="o", color="white", zorder=3, s=mpl.rcParams[
        #     "lines.markersize"] * 10)
        #ax.hlines(xaxis, class_quartiles[:, 0], class_quartiles[:, 2],
        #          color="black", zorder=1)

        # Plot outliers
        for x, outliers in zip(xaxis, outlier_data):
            ax.scatter(outliers, [x] * len(outliers), color="k")

        ax.set_xlabel(y_label)
        ax.set_yticks(xaxis)
        ax.set_yticklabels(x_labels)
        if yticks is not None:
            ax.set_xticks(yticks)

    fig.tight_layout()
    if figname:
        save_fig(fig, figname, **kwargs)

    return fig


def multi_hist(df, column_list, xlabel, ylabel, n_cols=2, transform=None, sharex=True, sharey=True, bins=10,
               pseudocount=0, figname=None, big_dimensions=True):
    """Make a figure with multiple subplots, each subplot containing a histogram for a different column of the
    dataframe. Optionally add a pseudocount and transform the data before plotting.

    Parameters
    ----------
    df : pd.DataFrame
        The data to plot
    column_list : list-like
        Column names to plot. Each column is plotted on a separate histogram.
    xlabel : str
        Label for the x-axis of the plots
    ylabel : str
        Label for the y-axis of the plots
    n_cols : int
        Number of columns in the multiplot
    transform : function handle
        If specified, add a pseudocount to the data and then apply the transformation function.
    sharex : bool
        Indicates if the x-axis should be shared across subplots.
    sharey : bool
        Same as sharex for y-axis.
    bins : int
        Number of bins for the histogram.
    pseudocount : int or float
        Add a pseudocount to the data if a transformation function is specified.
    figname : str
        If specified, save the figure with this name.
    big_dimensions : bool
        If True, then the size of the multiplot is the default figure size multiplied by the number of rows/columns.
        If False, then the entire figure is the default figure size.

    Returns
    -------
    fig : Figure handle
    """
    n_plots = len(column_list)
    fig, ax_list = setup_multiplot(n_plots, n_cols=n_cols, sharex=sharex, sharey=sharey, big_dimensions=big_dimensions)
    if len(ax_list.shape) == 1:
       ax_list = np.reshape(ax_list, (len(ax_list), 1)) 
    
    n_rows, _ = ax_list.shape # Used for the x axis display

    for i in range(n_plots):
        row, col = np.unravel_index(i, ax_list.shape)
        ax = ax_list[row, col]
        label = column_list[i]

        # Get rid of any NaN in the data since this is different from a zero
        data = df[label]
        data = data[data.notna()]

        if transform:
            data = transform(data + pseudocount)

        ax.hist(data, bins)
        ax.set_title(label)

        # Add axis labels if the axis is not shared or the axis is shared and on the appropriate axis.
        if not sharex or row == n_rows - 1:
            ax.set_xlabel(xlabel)
        if not sharey or col == 0:
            ax.set_ylabel(ylabel)

    if figname:
        save_fig(fig, figname, tight_layout=True)

    return fig


def roc_pr_curves(xaxis, tpr_list, precision_list, model_names, model_colors=None, prc_chance=None,
                  prc_upper_ylim=None, figname=None, legend=True, figax=None, **kwargs):
    """Make a ROC and PR curve for each model, optionally with a SD. Compute an AUC score for each curve.

    Parameters
    ----------
    xaxis : list-like
        The FPR and Recall, i.e. the x-axis for both plots. All TPR and Precision lists should be
        interpolated/computed to reflect the values at each point on xaxis.
    tpr_list : list of lists, shape = [n_models, len(xaxis)]
        tpr_list[i] corresponds to the TPR values for model i along xaxis. If tpr_list[i] is a list, then do not plot a
        standard deviation of the TPR. If tpr_list[i] is a list of lists, then it represents the TPR of each fold
        from cross-validation, in which case it is used to compute the mean and std of the TPR.
    precision_list : list of lists, shape = [n_models, len(xaxis)]
        precision_list[i] corresponds to the precision values for model i along xaxis. If precision_list[i] is a list,
        then do not plot a standard deviation of the precision. If precision_list[i] is a list of lists,
        then it represents the precision of each fold from cross-validation, in which case it is used to compute the
        mean and std of the precision.
    model_names : list-like
        The name of each model.
    model_colors : list-like or None
        If not none, the color to use for each model.
    prc_chance : float or None
        If not none, plot a chance line for the PR curve at this value.
    prc_upper_ylim : float or None
        If specified, the upper ylim for the PR curve. Otherwise, use the uper ylim of the ROC curve.
    figname : str or None
        If specified, save the figure with prefix figname.
    legend : bool
        If specified, display a legend.
    figax : ([figure, figure], [axes, axes]) or None
        If specified, make the plot in the two provided axes. Otherwise, generate a new axes.
    kwargs : dict
        Additional parameters for saving a figure.

    Returns
    -------
    fig_list : The handle to both figures (one for the ROC and one for the PR).
    auroc_list : AUROC scores for each model
    auroc_std_list : 1SD of AUROC scores for each model, or None if not computed.
    aupr_list : AUPR scores for each model
    aupr_std_list : 1SD of AUPR scores for each model, or None if not computed.

    """
    if figax:
        fig_list, ax_list = figax
    else:
        fig_roc, ax_roc = plt.subplots()
        fig_pr, ax_pr = plt.subplots()
        fig_list = [fig_roc, fig_pr]
        ax_list = [ax_roc, ax_pr]

    # If no colors specified, evenly sample the colormap to color each model
    if model_colors is None:
        model_colors = np.linspace(0, 0.99, len(model_names))
        model_colors = set_color(model_colors)

    # ROC curves
    ax = ax_list[0]
    auroc_list, auroc_std_list = _plot_each_model(ax, xaxis, tpr_list, model_colors, model_names)

    # Chance line
    ax.plot(xaxis, xaxis, color="black", linestyle="--", zorder=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_aspect("equal")
    if legend:
        ax.legend(loc="lower right", frameon=False)

    # ylim of ROC curve will help format PR curve
    lower_ylim, upper_ylim = ax.get_ylim()

    # PR curves
    ax = ax_list[1]
    aupr_list, aupr_std_list = _plot_each_model(ax, xaxis, precision_list, model_colors, model_names)

    # Optional chance line and formatting
    if prc_chance:
        ax.axhline(prc_chance, color="black", linestyle="--", zorder=1)
    if not prc_upper_ylim:
        prc_upper_ylim = upper_ylim
    ax.set_ylim(bottom=lower_ylim, top=prc_upper_ylim)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_aspect("equal")
    if legend:
        ax.legend(frameon=False)

    if figname:
        save_fig(fig_list[0], figname + "Roc", **kwargs)
        save_fig(fig_list[1], figname + "Pr", **kwargs)

    return fig_list, auroc_list, auroc_std_list, aupr_list, aupr_std_list


def _plot_each_model(ax, xaxis, y_list, model_colors, model_names):
    """Helper function for roc_pr_curves to plot each model on an Axes object.

    """
    area_list = []
    area_std_list = []
    for y, color, name in zip(y_list, model_colors, model_names):
        y = np.array(y)
        area_std = None

        # If y is a list of lists (i.e. a matrix), then compute the std of the curve and AUC
        if len(y.shape) == 2:
            y_std = np.std(y, axis=0)
            # Compute std of AUC and format as a string
            area_std = np.std([auc(xaxis, i) for i in y])

            # Now compute the mean curve
            y = y.mean(axis=0)

            # The std can't go above 1 or below 0
            y_std_upper = np.min([y + y_std, np.ones(y.size)], axis=0)
            y_std_lower = np.max([y - y_std, np.zeros(y.size)], axis=0)

            # Plot the std of the curve
            ax.fill_between(xaxis, y_std_lower, y_std_upper, alpha=0.2, zorder=2, color=color)

        # Plot the curve and compute AUC
        area = auc(xaxis, y)
        ax.plot(xaxis, y, label=name, zorder=3, color=color)
        area_list.append(area)
        area_std_list.append(area_std)

    return area_list, area_std_list


def stacked_bar_plots(df, ax_name, group_names, value_colors, legend_upper_left=None, legend_title=None,
                      legend_cols=1, vert=False, plot_title=None, figname=None, figax=None, **kwargs):
    """Make stacked bar plots, one bar per row of the provided DataFame, and optionally show a legend.

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot, rows are bar groups, columns are different values/colors
    ax_name : str
        Name of the axis for the plot
    group_names : list[str]
        Names of each group to display as ticks
    value_colors : list-like, length = len(df.columns)
        Color for each value of the df
    legend_upper_left : tuple(float, float)
        If specified, make a legend, with the upper left corner of the bounding box at these axes coordinates.
    legend_title : str
        If specified, title for the legend.
    legend_cols : int
        If specified, number of columns for the legend. Default is 1.
    vert : bool
        If False (default), make a horizontal bar plot. If True, make a vertical bar plot.
    plot_title : str
        If specified, title for the plot.
    figname : str
        If specified, save the figure to this filename.
    figax : (figure, axes) or None
        If specified, make the plot in the provided axes. Otherwise, generate a new axes.
    kwargs : for save_fig

    Returns
    -------
    fig : Figure handle
    """
    tick_values = np.arange(len(group_names))
    margin_edge = np.zeros(len(tick_values))
    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    for (label, values), color in zip(df.items(), value_colors):
        if vert:
            ax.bar(tick_values, values, color=color, label=label, bottom=margin_edge, tick_label=group_names)
        else:
            ax.barh(tick_values, values, color=color, label=label, left=margin_edge, tick_label=group_names)

        # Advance the margin
        margin_edge += values

    # Set the max of axis
    if vert:
        ax.set_ylim(top=margin_edge.max())
    else:
        ax.set_xlim(right=margin_edge.max())

    # Add axis label
    if vert:
        ax.set_ylabel(ax_name)
    else:
        ax.set_xlabel(ax_name)

    # Add legend if specified
    if legend_upper_left:
        legend_args = {"ncol": legend_cols, "bbox_to_anchor": legend_upper_left, "loc": "upper left"}
        if legend_title:
            legend_args["title"] = legend_title
        ax.legend(**legend_args)

    if plot_title:
        ax.set_title(plot_title)

    if figname:
        save_fig(fig, figname, **kwargs)

    return fig


def annotate_heatmap(ax, df, thresh, adjust_lower_triangle=False):
    """Display numbers on top of a heatmap to make it easier to view for a reader. If adjust_lower_triangle is True,
    then the lower triangle of the heatmap will display values in parentheses. This should only happen if the heatmap
    is symmetric. Assumes that low values are displayed as a light color and high values are a dark color.

    Parameters
    ----------
    ax : Axes object
        The plot containing the heatmap on which annotations should be made
    df : pd.DataFrame
        The data underlying the heatmap.
    thresh : float
        Cutoff for switching from dark to light colors. Values above the threshold will be displayed as white text,
        those below as black text.
    adjust_lower_triangle : bool
        If True, the lower triangle values will be shown in parentheses.

    Returns
    -------
    None
    """
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            value = df.iloc[row, col]
            if value > thresh:
                color = "white"
            else:
                color = "black"

            # Format the value as text
            value = f"{value:.2f}"
            # Add parentheses if desired and in the lower triangle and the heatmap is square
            if adjust_lower_triangle and row < col and df.shape[0] == df.shape[1]:
                value = "(" + value + ")"

            ax.text(row, col, value, ha="center", va="center", color=color)


# LEGACY FUNCTION
def gkmsvm_best_kmers(kmer_scores, positives, negatives, num_kmers=500, center_width=0):
    """Generate a plot to visualize the frequency and location of the best scoring k-mers. All positive and negative
    sequences are searched for the num_kmers most positively weighted k-mers and the num_kmers most negatively
    weighted k-mers. Then, count the number of times each k-mer occurs at each position of the sequences. Make a plot
    with the k-mer position on the x axis, k-mer weight on the y axis, the number of sequences with that k-mer at
    that position indicated by the size of the circle, and the classes of sequences specified by the color of the
    circle. Note that this function assumes all sequences are the same length.

    Parameters
    ----------
    kmer_scores : pd.Series
        The scores assigned to every k-mer
    positives : pd.Series
        All sequences belonging to the positives. If positives.name is specified, it is used in creating the figure
        legend.
    negatives : pd.Series
        All sequences belonging to the negatives. If negatives.name is specified, it is used in creating the figure
        legend.
    num_kmers : int
        The number of top scoring k-mers to analyze. Both the num_kmers most positive and num_kmers most negatives
        k-mers are analyzed, i.e. 2*num_kmers are analyzed.
    center_width : int
        If greater than 1, draw a vertical grey rectangle along the center of the x axis.

    Returns
    -------
    fig : The figure handle.

    """
    # Easy way to get two slices of the Series
    kmer_scores = kmer_scores.sort_values()
    best_kmers = kmer_scores[np.r_[:num_kmers, -num_kmers:0]]

    # Assumes all sequences are the same length. We want to plot the position of the k-mer *center*, not the start of
    #  the k-mer.
    seq_len = len(positives.iloc[0])
    kmer_len = len(kmer_scores.index[0])
    shift_factor = int((seq_len - kmer_len) / 2)
    positive_positions_df = _count_top_kmers(best_kmers, positives, shift_factor)
    negative_positions_df = _count_top_kmers(best_kmers, negatives, shift_factor)

    row_size, col_size = mpl.rcParams["figure.figsize"]
    marker_size = mpl.rcParams["lines.markersize"]
    if positives.name:
        positive_label = positives.name
    else:
        positive_label = "Positives"
    if negatives.name:
        negative_label = negatives.name
    else:
        negative_label = "Negatives"

    # Get the range for the positive and negative k-mers so that the subplots are sized appropriate
    smallest_pos = best_kmers[best_kmers > 0].min()
    smallest_neg = best_kmers[best_kmers < 0].max()
    pos_range = best_kmers[best_kmers > 0].max() - smallest_pos
    neg_range = smallest_neg - best_kmers[best_kmers < 0].min()

    fig, (ax_pos, ax_neg) = plt.subplots(nrows=2, ncols=1, figsize=(row_size * 1.5, col_size), gridspec_kw={
        "height_ratios": [pos_range, neg_range]})

    # First plot everything on both axes. Then, we will resize the axes so ax_pos only shows the positive k-mers and
    # ax_neg only shows the negative k-mers.
    points_neg = ax_neg.scatter(negative_positions_df["Position"], negative_positions_df["Weight"],
                                s=marker_size * negative_positions_df["Count"], alpha=0.25,
                                label=negative_label, color="blue", zorder=2)
    points_pos = ax_neg.scatter(positive_positions_df["Position"], positive_positions_df["Weight"],
                                s=marker_size * positive_positions_df["Count"], alpha=0.25,
                                label=positive_label, color="red", zorder=2)
    points_neg = ax_pos.scatter(negative_positions_df["Position"], negative_positions_df["Weight"],
                                s=marker_size * negative_positions_df["Count"], alpha=0.25,
                                label=negative_label, color="blue", zorder=2)
    points_pos = ax_pos.scatter(positive_positions_df["Position"], positive_positions_df["Weight"],
                                s=marker_size * positive_positions_df["Count"], alpha=0.25,
                                label=positive_label, color="red", zorder=2)

    # Make the rectangle if desired
    bottom, top = ax_pos.get_ylim()
    if center_width > 1:
        rect_start = int(-center_width / 2)
        rect = mpatches.Rectangle((rect_start, bottom), center_width, top - bottom, color="grey", alpha=0.35, zorder=1)
        ax_pos.add_patch(rect)
        rect = mpatches.Rectangle((rect_start, bottom), center_width, top - bottom, color="grey", alpha=0.35, zorder=1)
        ax_neg.add_patch(rect)

    # Trim the y axes
    extra_whitespace = 0.98
    ax_pos.set_ylim(smallest_pos * extra_whitespace, top)
    ax_neg.set_ylim(bottom, smallest_neg * extra_whitespace)

    # Hide the spines between the axes
    ax_neg.spines["top"].set_visible(False)
    ax_pos.spines["bottom"].set_visible(False)
    ax_pos.tick_params(axis="x", bottom=False, labelbottom=False)

    # Get unique values of marker sizes and select 5 to display in the legend
    uniq_point_sizes = np.unique(np.concatenate((points_pos.get_sizes(), points_neg.get_sizes())), axis=None)
    uniq_point_sizes = uniq_point_sizes[np.linspace(0, uniq_point_sizes.size - 1, 5).round().astype(int)]
    class_handles = [points_pos, points_neg]
    point_handles = []
    for i in uniq_point_sizes:
        point_size = np.sqrt(i)
        label = int(i / marker_size)
        handle = mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=point_size,
                               label=f"{label}")
        point_handles.append(handle)

    # Axis labels
    ax_neg.set_xlabel("Center of k-mer Relative\nto Center of Sequence")
    fig.text(0.05, 0.5, "k-mer Weight", ha="center", va="center", rotation="vertical",
             fontsize=mpl.rcParams["axes.labelsize"])

    # Legends
    legend_font = mpl.rcParams["legend.fontsize"]
    legend = ax_pos.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), handles=class_handles, title="Sequence Class")
    plt.setp(legend.get_title(), fontsize=legend_font)
    legend = ax_neg.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), handles=point_handles, title="Number of "
                                                                                                     "Sequences")
    plt.setp(legend.get_title(), fontsize=legend_font)

    # Add hatch marks
    diag_size = 0.015
    diag_kwargs = dict(transform=ax_pos.transAxes, color="black", clip_on=False)
    # Top left
    ax_pos.plot((-diag_size, diag_size), (-diag_size, diag_size), **diag_kwargs)
    # Top right
    ax_pos.plot((1 - diag_size, 1 + diag_size), (-diag_size, diag_size), **diag_kwargs)

    diag_kwargs.update(transform=ax_neg.transAxes)
    diag_y_scaler = pos_range / neg_range
    # Bottom left
    ax_neg.plot((-diag_size, diag_size), (1 - diag_size * diag_y_scaler, 1 + diag_size * diag_y_scaler), **diag_kwargs)
    # Bottom right
    ax_neg.plot((1 - diag_size, 1 + diag_size), (1 - diag_size * diag_y_scaler, 1 + diag_size * diag_y_scaler), **diag_kwargs)

    fig.tight_layout(rect=(0.1, 0, 1, 1))
    return fig


def _count_top_kmers(best_kmers, sequences, shift_factor):
    """Helper function for gkmsvm_best_kmers to identify the best k-mers in a set of sequences and determine the
    number of times the k-mer occurs at each position of the sequences.

    Parameters
    ----------
    best_kmers : pd.Series
        The k-mers to analyze, index is the k-mer and value is the weight.
    sequences : pd.Series
        The sequences to use to search for the k-mers.
    shift_factor : int
        Value to subtract from k-mer position to get the position of the center of the k-mer relative to the center
        of the sequences.

    Returns
    -------
    position_counts_df : pd.DataFrame
        Each row contains the sequence position, k-mer weight, and number of sequences with that k-mer at that position

    """
    position_counts = []
    for kmer, weight in best_kmers.iteritems():
        kmer_pos = sequences.str.find(kmer)
        # Anything that has a position of -1 is missing, fill it with an nan.
        kmer_pos[kmer_pos == -1] = np.nan
        kmer_pos -= shift_factor
        kmer_pos = kmer_pos.value_counts()
        for position, counts in kmer_pos.iteritems():
            position_counts.append([position, weight, counts])

    position_counts_df = pd.DataFrame(position_counts, columns=["Position", "Weight", "Count"])
    return position_counts_df

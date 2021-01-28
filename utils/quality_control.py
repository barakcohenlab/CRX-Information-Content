"""
Author: Ryan Friedman (@rfriedman22)
Email: ryan.friedman@wustl.edu
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import modeling
import plot_utils


def read_bc_count_files(files, labels):
    """Read in barcode count files and store the result in a DataFrame, where rows are barcodes and columns are
    different replicates/experiments/conditions.

    Parameters
    ----------
    files : list-like
        File names, each of which contains barcode counts from different replicates/experiments/conditions.
    labels : list-like
        Name to assign to each file, becomes the columns of the DataFrame.

    Returns
    -------
    bc_counts_df : pd.DataFrame
        Raw barcode counts across multiple conditions.

    """
    # Use the first file to get the barcodes and the sequence label ID
    bc_counts_df = pd.read_csv(files[0], sep="\t", index_col=0, usecols=["barcode", "label"])

    # Then for each file, read in the barcodes and counts, then join with the label ID
    for file, name in zip(files, labels):
        counts = pd.read_csv(file, sep="\t", index_col=0, usecols=["barcode", "count"], squeeze=True)
        bc_counts_df[name] = counts

    return bc_counts_df


def per_million(df):
    return df.sum() / 10**6


def assess_balance_coverage(df, sample_labels):
    """Compute the total number of barcode counts, plot the coverage per barcode in each sample, and plot the
    percentage of barcode counts represented by each sample.

    Parameters
    ----------
    df : pd.DataFrame
        Barcode counts in multiple conditions/samples. Rows are barcodes, columns are the samples.
    sample_labels : list-lie
        Column names to use in assessing balance and coverage

    Returns
    -------
    Nothing

    """
    total_counts_df = df[sample_labels].sum()
    n_barcodes = df.index.size
    print(f"There are a total of {per_million(total_counts_df): .3f} million barcode counts.")

    # Plot the coverage
    xaxis = np.arange(len(sample_labels))
    fig, ax = plt.subplots()
    ax.bar(xaxis, total_counts_df / n_barcodes)
    ax.set_ylabel("Coverage per barcode")
    ax.set_xticks(xaxis)
    ax.set_xticklabels(sample_labels)

    # Plot the balance
    fig, ax = plt.subplots()
    ax.bar(xaxis, total_counts_df / total_counts_df.sum() * 100)
    ax.set_ylabel("Pct. total counts")
    ax.set_xticks(xaxis)
    ax.set_xticklabels(sample_labels)


def filter_low_counts(df, sample_labels, cutoff_values, dna_labels=None, bc_per_seq=3, n_rna_samples_missing=0,
                      n_dna_samples_missing=0, cpm_normalize=True):
    """Filter out barcodes that are less than the provided thresholds for each sample. If the barcode is missing from
    DNA, then replace it with an NaN in all samples. If it is only missing from RNA, then replace it with a zero in
    all samples. Then, normalize by counts per million for each sample.

    Parameters
    ----------
    df : pd.DataFrame
        Raw barcode counts.
    sample_labels : list-like
        Labels of the df indiciating which columns contain barcode counts.
    cutoff_values : list-like
        Cutoff threshold for each sample. Barcodes with counts less than the threshold in any sample are removed.
    dna_labels : list-like
        If specified, then print status reports of how many barcodes and sequences are lost in DNA samples.
    bc_per_seq : int
        The number of barcodes per sequence, used to determine how many sequences are lost due to DNA drop-out.
    n_rna_samples_missing : int
        Number of samples that can be missing a barcode before it is replaced with an NaN in all samples. E.g. if
        n_samples_missing is 0, the barcode must pass the cutoff in every replicate to be maintained. If it is 1,
        then the barcode must pass the cutoff in all but one sample to be maintained. If n_samples_missing > 0,
        then barcodes are only replaced with NaN in the replicates where they are below the cutoff, but if that
        barcode is missing from at least n_samples_missing samples, it is replaced with an NaN in all samples.
    n_dna_samples_missing : int
        Same as n_rna_samples_missing but for DNA.
    cpm_normalize : bool
        If True, normalize each sample to CPM after thresholding. Otherwise, simply return the thresholded counts.

    Returns
    -------
    filtered_df : pd.DataFrame
        The filtered barcode counts, normalized to counts per million.
    """
    # Store the cutoffs as a series, labels as index and cutoffs as values
    df = df.copy()
    cutoffs = pd.Series({i: j for i, j in zip(sample_labels, cutoff_values)})

    # If there are DNA samples, first filter any barcodes missing from the DNA.
    if dna_labels is not None:
        dna_mask = df.apply(
            lambda x: pd.Series({i: x[i] < j for i, j in cutoffs[dna_labels].iteritems()}),
            axis=1
        )
        print("Barcodes missing in DNA:")
        for label in dna_labels:
            print(f"Sample {label}: {dna_mask[label].sum()} barcodes")
            # Replace barcodes below their cutoff with an NaN
            df.loc[dna_mask[label], label] = np.nan

        # Drop any barcodes missing from too many DNA samples
        dna_missing_mask = dna_mask.sum(axis=1) > n_dna_samples_missing
        print(f"{dna_missing_mask.sum()} barcodes are missing from more than {n_dna_samples_missing} DNA samples.")
        df.loc[dna_missing_mask, dna_labels] = np.nan

        rna_labels = sample_labels[~pd.Series(sample_labels).isin(dna_labels)]

    else:
        rna_labels = sample_labels

    # Filter barcodes missing from RNA
    rna_mask = df.apply(
        # For each row, determine if the column is above the cutoff
        lambda x: pd.Series({i: x[i] < j for i, j in cutoffs[rna_labels].iteritems()}),
        axis=1
    )
    print("Barcodes off in RNA:")
    for label in rna_labels:
        print(f"Sample {label}: {rna_mask[label].sum()} barcodes")

    # Drop barcodes missing from too many RNA samples
    rna_missing_mask = rna_mask.sum(axis=1) > n_rna_samples_missing
    print(f"{rna_missing_mask.sum()} barcodes are off in more than {n_rna_samples_missing} RNA samples.")
    df.loc[rna_missing_mask, rna_labels] = 0

    # Normalize the data to CPM if desired
    if cpm_normalize:
        assess_balance_coverage(df, sample_labels)
        df[sample_labels] /= per_million(df[sample_labels])

    return df


def reproducibility_plots(df, labels, unit, big_dimensions=False):
    """Make a multiplot of scatters comparing pairs of replicates to assess reproducibility.

    Parameters
    ----------
    df : pd.DataFrame
        Rows are samples, columns are different replicates.
    labels : list-like
        Column names representing replicates.
    unit : str
        Unit of the counts (raw, cpm, RNA/DNA, etc.)
    big_dimensions : bool
        Indicates whether or not the reproducibility plot should use big dimensions

    Returns
    -------
    fig : figure handle
    """
    text_size = mpl.rcParams["axes.labelsize"]

    n_samples = len(labels)
    fig, ax_list = plot_utils.setup_multiplot(n_samples**2, n_cols=n_samples, sharex=True, sharey=True,
                                              big_dimensions=big_dimensions)
    for i in range(ax_list.size):
        row, col = np.unravel_index(i, ax_list.shape)
        ax = ax_list[row, col]

        # If on the diagonal, display the name of the replicate
        if row == col:
            ax.text(0.5, 0.5, f"{labels[row]} {unit}", fontsize=text_size,
                    transform=ax.transAxes, ha="center", va="center")
        else:
            x = df[labels[col]]
            y = df[labels[row]]

            # Display the scatter plots above the diagonal
            if row < col:
                ax.scatter(x, y, c="black")
            # Display the R**2 below the diagonal
            else:
                pcc = x.corr(y)
                ax.text(0.5, 0.5, f"$R^2 = ${pcc**2: 0.3f}", fontsize=text_size,
                        transform=ax.transAxes, ha="center", va="center")

    return fig


def normalize_rna_by_dna(df, rna_labels, dna_labels, average_dna=False):
    """Normalize RNA counts by DNA counts and return the result.

    Parameters
    ----------
    df : pd.DataFrame
        Contains columns for both RNA and DNA replicates, and potentially other information too.
    rna_labels : list-like
        Column names for RNA labels.
    dna_labels : list-like
        Column names for DNA labels. If there are multiple, assumes that ordering is matched with RNA labels.
    average_dna : bool
        If True, take the mean barcode count across DNA replicates before normalizing RNA counts. If False,
        then normalize RNA counts by its paired DNA sample.

    Returns
    -------
    norm_df : pd.DataFrame
        The original df, but RNA columns normalized to their DNA counterparts.
    """
    norm_df = df.copy()
    dna_labels = _check_rna_dna_labels(rna_labels, dna_labels)

    if average_dna:
        dna_counts = norm_df[dna_labels].mean(axis=1)
        for rna in rna_labels:
            norm_df[rna] /= dna_counts
    else:
        for dna, rna in zip(dna_labels, rna_labels):
            norm_df[rna] /= norm_df[dna]

    return norm_df


def _check_rna_dna_labels(rna, dna):
    """
    Make sure there is only one DNA label or the same number of DNA labels as RNA labels. Both arguments must be
    list-like. Returns an ndarray of DNA labels for normalization.
    """
    if len(dna) == 1:
        dna = np.full(len(rna), dna[0])

    if len(dna) != len(rna):
        raise Exception("The number of DNA labels is neither 1 nor the number of RNA labels. Cannot normalize data.")

    return dna


def bc_counts_vs_dna_plot(df, rna_labels, dna_labels, n_cols=2, big_dimensions=False, y_label="RNA/DNA Counts"):
    """Plot RNA/DNA counts vs. DNA counts to make sure RNA/DNA indicates activity and is not an artifact of DNA counts.

    Parameters
    ----------
    df : pd.DataFrame
        Contains columns for both RNA and DNA replicates, and potentially other information too.
    rna_labels : list-like
        Column names for RNA labels.
    dna_labels : list-like
        Column names for DNA labels. If there are multiple, assumes that ordering is matched with RNA labels.
    n_cols : int
        Number of columns for the multiplot.
    big_dimensions : bool
        Indicates whether or not the reproducibility plot should use big dimensions
    y_label : str
        Label for the y axis

    Returns
    -------
    fig : figure handle
    """
    dna_labels = _check_rna_dna_labels(rna_labels, dna_labels)
    fig, ax_list = plot_utils.setup_multiplot(len(rna_labels), n_cols=n_cols, big_dimensions=big_dimensions)
    text_size = mpl.rcParams["axes.labelsize"]

    # For each RNA sample, plot RNA/DNA on the y and DNA on the x
    for i, (dna, rna) in enumerate(zip(dna_labels, rna_labels)):
        ax = ax_list[np.unravel_index(i, ax_list.shape)]
        x = df[dna]
        y = df[rna]
        pcc = x.corr(y)**2

        ax.scatter(x, y, color="k")
        ax.text(0.95, 0.95, f"$R^2 = ${pcc: 0.3f}", transform=ax.transAxes, fontsize=text_size, ha="right", va="center")
        ax.set_xlabel("DNA CPM")
        ax.set_ylabel(y_label)
        ax.set_title(rna)

    return fig


def _make_basal_mask(df, key):
    return df["label"].str.contains(key, case=False)


def average_barcodes(df, sequence_label="label", out_prefix=None):
    """Average RNA/DNA barcode counts for each sequence within replicates.

    Parameters
    ----------
    df : pd.DataFrame
        Index is the barcode, one column must have the key sequence_label, the rest are assumed to be RNA/DNA counts
        for each replicate.
    sequence_label : str
        Name of the column in df containing the sequence IDs
    out_prefix : str
        If specified, save the df to file with this prefix.

    Returns
    -------
    expression_df : pd.DataFrame
        Average RNA/DNA counts for each sequence in each replicate. Index is now the sequence label, column is the
        replicate.
    """
    expression_df = df.groupby(sequence_label).mean()
    if out_prefix:
        expression_df.to_csv(f"{out_prefix}AverageExpressionPerReplicate.txt", sep="\t", na_rep="NaN")

    return expression_df


def basal_normalize(df, basal_key):
    """Normalize activity levels to basal in the replicate, then take averages and stddev across replicates.

    Parameters
    ----------
    df : pd.DataFrame
        Index is sequence ID, columns are average RNA/DNA barcode counts for each replicate.
    basal_key : str
        Index value for basal.

    Returns
    -------
    activity_df : pd.DataFrame
        Index is sequence ID, columns are the basal-normalized mean, stddev, and number of replicates the sequence was
        measured in.
    """
    activity_df = df / df.loc[basal_key]
    # Drop basal
    activity_df = activity_df.drop(index=basal_key)

    # Convert basal-normalized levels to summary statistics
    activity_df = activity_df.apply(lambda x: pd.Series({"expression": x.mean(), "expression_std": x.std(),
                                                         "expression_reps": x.count()}),
                                    axis=1)
    return activity_df


def log_ttest_vs_basal(df, basal_key):
    """Do t-tests in log space to see if sequences has the same activity as basal.

    Parameters
    ----------
    df : pd.DataFrame
        Index is sequence ID, columns are average RNA/DNA barcode counts for each replicate.
    basal_key : str
        Index value for basal.

    Returns
    -------
    pvals : pd.Series
        p-value for t-test of the null hypothesis that the log activity of a sequence is the same as that of basal.
        Does not include a p-value for basal.
    """
    log_params = df.apply(_get_lognormal_params, axis=1)

    # Pull out basal params
    basal_mean, basal_std, basal_n = log_params.loc[basal_key]

    # Drop basal from the df
    log_params = log_params.drop(index=basal_key)

    # Do t-tests on each row
    pvals = log_params.apply(lambda x: stats.ttest_ind_from_stats(basal_mean, basal_std, basal_n,
                                                                  x["mean"], x["std"], x["n"],
                                                                  equal_var=False)[1],
                             axis=1)
    return pvals


def _get_lognormal_params(row):
    """Helper function to get parameters of lognormal distribution from linear data.

    Parameters
    ----------
    row : pd.Series
        Row of a df corresponding to barcode averages in each replicate.

    Returns
    -------
    params : pd.Series
        mu and sigma for the lognormal distribution, and the number of replicates the sequence was measured in.
    """
    mean = row.mean()
    std = row.std()
    cov = std / mean

    # Rely on the fact that the mean is exp(mu + 1/2 sigma**2) and the variance is mean**2 * (exp(sigma**2) - 1)
    log_mean = np.log(mean / np.sqrt(cov**2 + 1))
    log_std = np.sqrt(np.log(cov**2 + 1))
    params = pd.Series({
        "mean": log_mean,
        "std": log_std,
        "n": row.count()
    })

    return params


def compare_to_normal(data, xlabel, bins=10, show_mean=True, show_median=True, figname=None):
    """Given some data, Z-score it and perform a KS test against the normal distribution. Then plot a histogram of
    the data.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame with one column
        Values of the data of interest.
    xlabel : str
        Label for the x-axis.
    bins : int
        Number of bins for the histogram.
    show_mean : bool
        If True, plot the mean on the histogram.
    show_median : bool
        If True, plot the median on the histogram.
    figname : str
        If specified, save the figure using this name.

    Returns
    -------
    fig : Figure handle
    pval : float
        The p-value from a KS test of the Z-scored data, where the null hypothesis is that the data is normally
        distributed.
    """
    zscores = stats.zscore(data)

    # Are the sequences normally distributed?
    _, pval = stats.kstest(zscores, cdf="norm")

    # Make the histogram
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins, density=True, label=None)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")

    # Put the p-value in the title
    ax.set_title(f"KS test to normal p={pval : 1.2e}")

    if show_mean:
        mean = data.mean()
        ax.axvline(mean, color="k", label=f"mean={mean: .2f}")

    if show_median:
        median = data.median()
        ax.axvline(median, color="k", linestyle="--", label=f"median={median: .2f}")

    ax.legend()

    if figname:
        plot_utils.save_fig(fig, figname)

    return fig, pval


def compare_two_sets(data1, data2, xlabel, label1, label2, bins=10, show_means=True, show_medians=True,
                     figname=None):
    """Perform a 2-sample KS test with the null hypothesis that the two datasets come from the same distribution.
    Then, plot a histogram with the two datasets.

    Parameters
    ----------
    data1 : pd.Series or pd.DataFrame with one column
        Values for the first dataset.
    data2 : pd.Series or pd.DataFrame with one column
        Values for the second dataset.
    xlabel : str
        Label for the x-axis of the histogram.
    label1 : str
        Name of the first dataset.
    label2 : str
        Name of the second dataset.
    bins : int
        Number of bins for the histogram.
    show_means : bool
        If True, put the means of the two datasets on the histogram.
    show_medians : bool
        If True, put the medians of the two datasets on the histogram.
    figname : str
        If specified, save the figure with this name.

    Returns
    -------
    fig : Figure handle
    pval : float
        The p-value from a KS test of the Z-scored data, where the null hypothesis is that the data is normally
        distributed.
    """
    # Perform the 2-sample KS test
    _, pval = stats.ks_2samp(data1, data2)

    fig, ax = plt.subplots()
    ax.hist(data1, bins=bins, label=label1, alpha=0.5, density=True, color=plot_utils.set_color(0.25))
    ax.hist(data2, bins=bins, label=label2, alpha=0.5, density=True, color=plot_utils.set_color(0.75))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(f"2-sample KS test p={pval: 1.2e}")

    # Display the means if desired
    if show_means:
        ax.axvline(data1.mean(), color="k", label=label1 + " mean")
        ax.axvline(data2.mean(), color="k", linestyle="--", label=label2 + " mean")

    # Display medians if desired
    if show_medians:
        ax.axvline(data1.median(), color="k", linestyle=":", label=label1 + " median")
        ax.axvline(data2.median(), color="k", linestyle="-.", label=label2 + " median")

    ax.legend()
    if figname:
        plot_utils.save_fig(fig, figname)

    return fig, pval

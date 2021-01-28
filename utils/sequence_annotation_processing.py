"""
Author: Ryan Friedman (@rfriedman22)
Email: ryan.friedman@wustl.edu
"""

import warnings

import numpy as np
import pandas as pd
from pybedtools import BedTool
from scipy import stats

import fasta_seq_parse_manip
import modeling


def save_df(df, outfile):
    """Save a DataFrame to file."""
    df.to_csv(outfile, sep="\t", na_rep="NaN")


def remove_mutations_from_seq_id(df):
    """Remove the mutation identifier from the sequence ID so all that is left is coordinates and annotations."""
    mapper = lambda x: "_".join(x.split("_")[:2])
    if isinstance(df, pd.DataFrame):
        kwargs = {"axis": 0, "level": "label", "mapper": mapper}
    elif isinstance(df, pd.Series):
        kwargs = {"index": mapper}
    else:
        warnings.warn("Not a DataFrame or Series, doing nothing!")
        return df
    
    df = df.copy()
    df = df.rename(**kwargs)
    
    return df


def compute_ref_vs_alt(df, ref_string, alt_string, ref_suffix, alt_suffix, activity_prefix="expression",
                       comparison_prefix="ref_vs_alt"):
    """Compute effect sizes between pairs of sequences and which pairs are significantly different.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing activity measurements, statistics, and annotations. Index is formatted such that some
        IDs will contain the ref_string and others will contain alt_string.
    ref_string : str
        Identifier for reference sequence IDs.
    alt_string : str
        Identifier for alt sequence IDs.
    ref_suffix : str
        Suffix for columns containing info on the ref.
    alt_suffix : str
        Suffix for columns containing info on the alt.
    activity_prefix : str
        Prefix for columns containing activity measurements.
    comparison_prefix : str
        Prefix to use for ref/alt comparisons.

    Returns
    -------
    ref_alt_df : pd.DataFrame
        DataFrame containing ref and alt measurements merged onto the same row. The index is the "base" sequence
        identifier. Also contains the identifier/name for the ref and alt sequence so that, e.g. ref_alt_df.index +
        variant_<ref_suffix> corresponds to the label for the reference sequence.
    """
    df = df.copy()
    # Prepare to separate the data
    ref_mask = df.index.str.contains(ref_string)
    alt_mask = df.index.str.contains(alt_string)

    # Add the variant info as a column, and then trim it off the index
    df["variant"] = df.index.str.split("_").str[2:].str.join("_")
    df = remove_mutations_from_seq_id(df)

    # Rearrange the data so ref and alt are on the same rows
    ref_df = df[ref_mask]
    alt_df = df[alt_mask]

    ref_alt_df = ref_df.join(alt_df, lsuffix=ref_suffix, rsuffix=alt_suffix)

    # Compute fold change not in log2 space for statistics
    fold_change = ref_alt_df[f"{activity_prefix}{ref_suffix}"] - ref_alt_df[f"{activity_prefix}{alt_suffix}"]
    ref_alt_df[f"{comparison_prefix}_log2"] = ref_alt_df[f"{activity_prefix}_log2{ref_suffix}"] - \
                                              ref_alt_df[f"{activity_prefix}_log2{alt_suffix}"]

    # Do t-tests
    ref_sem_sq = ref_alt_df[f"{activity_prefix}_SEM{ref_suffix}"] ** 2
    alt_sem_sq = ref_alt_df[f"{activity_prefix}_SEM{alt_suffix}"] ** 2
    tstat = -(fold_change.abs()) / np.sqrt(ref_sem_sq + alt_sem_sq)
    ddof = (ref_sem_sq + alt_sem_sq) ** 2 / \
           ((ref_sem_sq ** 2 / (ref_alt_df[f"{activity_prefix}_num_barcodes{ref_suffix}"] - 1)) +
            (alt_sem_sq ** 2 / (ref_alt_df[f"{activity_prefix}_num_barcodes{alt_suffix}"] - 1)))
    ref_alt_df[f"{comparison_prefix}_pvalue"] = stats.t.cdf(tstat, ddof) * 2 # FIXME raises a warning due to NaN
    ref_alt_df[f"{comparison_prefix}_qvalue"] = modeling.fdr(ref_alt_df[f"{comparison_prefix}_pvalue"])

    return ref_alt_df



def compute_wt_vs_mut(df, mutation_matcher):
    warnings.warn("Using a deprecated function, use compute_ref_vs_alt instead to compute difference between "
                  "sequences.")
    df = df.copy()
    
    # Make the mutations a column rather than part of the index, if necessary
    if "mutations" in set(df.index.names):
        df = df.reset_index(level="mutations", col_level=1, col_fill="sequence")
    
    # Get sequences that have a full mutant
    has_mutant_mask = df["mutations"].str.contains(mutation_matcher)
    has_mutant_labels = df[has_mutant_mask].index.get_level_values("label")
    df = df[df.index.get_level_values("label").isin(has_mutant_labels)]
    
    # Now get sequences that also have a WT in the library
    wt_labels = df[df["mutations"] == "WT"].index.get_level_values("label")
    has_wt_mask = df.index.get_level_values("label").isin(wt_labels)
    df = df[has_wt_mask]
    
    # Separate out the WT and full MUT data
    wt_mask = (df["mutations"] == "WT")
    wt_df = df[wt_mask]
    full_mut_df = df[df["mutations"].str.contains(mutation_matcher)]
    
    # We have to do the WT vs MUT t-test manually because the raw data is not available at this stage. All we have is mean, SEM, and n.
    wt_expr = wt_df["expression"]
    wt_sem_sq = wt_df["expression_SEM"]**2
    wt_n = wt_df["expression_num_barcodes"]
    mut_expr = full_mut_df["expression"]
    mut_sem_sq = full_mut_df["expression_SEM"]**2
    mut_n = full_mut_df["expression_num_barcodes"]
    
    # Do t-tests. In order to get a proper p-value, wt_expr - mut_expr must be negative.
    comparison_prefix = "wt_vs_full_mut"
    tstat = -((wt_expr - mut_expr).abs())
    tstat = tstat / np.sqrt(wt_sem_sq + mut_sem_sq)
    ddof = (wt_sem_sq + mut_sem_sq)**2 / ((wt_sem_sq**2 / (wt_n - 1)) + (mut_sem_sq**2 / (mut_n - 1)))
    pvalues = stats.t.cdf(tstat, ddof) * 2 # FIXME raises a warning due to NaN
    pvalues = pd.Series(pvalues, index=ddof.index)
    df[f"{comparison_prefix}_pvalue"] = pvalues
    
    # Do FDR to correct for multiple hypotheses
    qvalues = modeling.fdr(pvalues, name_prefix=comparison_prefix)
    df = df.join(qvalues)

    # Compute log2(WT/MUT)
    df[f"{comparison_prefix}_log2_change"] = wt_df["expression_log2"] - full_mut_df["expression_log2"]
    
    return df


def get_nearest_tss(df, tss_file):
    """Annotate each sequence for the distance to the nearest TSS and the corresponding gene name.

    Parameters
    ----------
    df : pd.DataFrame
        The underlying data. Use chrom, start, stop to make a BedTool to find the nearest TSS, then add that
        information into the df.
    tss_file : str
        Name of BED file containing TSS and gene names.

    Returns
    -------
    closest : pd.DataFrame
        For each sequence label, indicates the distance to the nearest TSS and the gene name.
    """
    df = df.copy()
    # Since df is not necessarily sorted in BED order, add the label as a column so we can add the TSS information
    # back in correctly
    df["name"] = df.index.get_level_values("label")

    data_bed = BedTool.from_dataframe(df[["chrom", "start", "stop", "name"]]).sort()
    tss_bed = BedTool(tss_file).sort()
    # Resolve ties by picking the first TSS
    closest = data_bed.closest(tss_bed, t="first", d=True)
    closest = closest.to_dataframe(usecols=[3, 7, 10], index_col=0, names=["label", "nearest_tss_name",
                                                                           "nearest_tss_dist"])
    return closest


def to_categorical(data, categories=["Silencer", "Inactive", "Weak enhancer", "Strong enhancer"]):
    """Convert strings to pd.Categorical. By default, this is for activity bins, but it can be done for anything.

    Parameters
    ----------
    data : pd.Series[str]
        Strings containing the data to be converted into categorical dtype.
    categories : list[str]
        Names of the categories in data, listed in order so that categories[0] is the "lowest" value.

    Returns
    -------
    data : pd.Categorical
        Categorical representation of the data.
    """
    data = pd.Categorical(data, categories=categories, ordered=True)
    return data

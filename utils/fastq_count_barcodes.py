#!/usr/bin/env python3
"""
Map barcode counts to CRS labels and drop any barcodes not expected.
"""
# Author: Ryan Friedman
# Email: ryan.friedman@wustl.edu

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def main(prefix, bc_to_crs):
    logger.info(f"Counting barcodes for {prefix}...")
    putative_bc_count_df = f"{prefix}.tmp"
    # Read in BC counts
    putative_bc_count_df = pd.read_csv(putative_bc_count_df, sep="\t", header=None, names=["count", "barcode"],
                                       index_col=1, squeeze=True)
    # Read in mapping from BC to sequence ID
    bc_to_crs_df = pd.read_csv(bc_to_crs, sep="\t", header=None, names=["barcode", "label"], index_col=0, squeeze=True)

    # Join together the two DataFrames to determine barcode counts and representation
    bc_to_crs_df = pd.concat([bc_to_crs_df, putative_bc_count_df], axis=1, sort=True)
    total_read_counts = bc_to_crs_df["count"].sum()
    # Remove reads not in the barcode mapping
    bc_to_crs_df = bc_to_crs_df[bc_to_crs_df["label"].notnull()]
    bc_to_crs_df.index.names = ["barcode"]

    # Assess BC recovery
    n_bc = bc_to_crs_df.index.size
    represented_bc = bc_to_crs_df["count"].notna().sum() / n_bc * 100
    logger.info(f"{represented_bc: 2.2f}% of barcodes were recovered.")
    # Replace missing data with zeros
    bc_to_crs_df = bc_to_crs_df.fillna(value=0)
    total_bc_counts = bc_to_crs_df["count"].sum() / total_read_counts * 100
    logger.info(f"{total_bc_counts: 2.2f}% of reads had a barcode match.")
    # Save to file
    bc_to_crs_df.to_csv(f"{prefix}.counts", sep="\t", float_format="%d")

    # Make a histogram of the barcode counts
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist(np.log10(bc_to_crs_df["count"] + 1), bins=50, density=True)
    ax.set_xlabel("log10 Raw Barcode Count", fontsize=20)
    ax.set_ylabel("Density", fontsize=20)
    ax.set_title(prefix, fontsize=30)
    ax.tick_params(labelsize=15)
    fig.tight_layout()
    figname = f"{prefix}RawBarcodeCountHistogram"
    fig.savefig(f"{figname}.svg")
    fig.savefig(f"{figname}.png")


if __name__ == "__main__":
    # Setup console logging
    console = logging.StreamHandler()
    all_loggers = logging.getLogger()
    all_loggers.setLevel(logging.INFO)
    all_loggers.addHandler(console)
    log_format = "[%(asctime)s][%(levelname)-7s] %(message)s"
    log_formatter = logging.Formatter(log_format, "%Y-%m-%d %H:%M:%S")
    console.setFormatter(log_formatter)

    # Setup argument parsing
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "prefix", help="The prefix of the file with putative barcode counts and the prefix assigned to any output files"
    )
    parser.add_argument(
        "bc_to_crs", help="Filename that maps each barcode to the CRS identifier"
    )
    args = parser.parse_args()

    # Pass args to main
    main(args.prefix, args.bc_to_crs)

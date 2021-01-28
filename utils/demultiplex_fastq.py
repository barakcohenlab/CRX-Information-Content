#!/usr/bin/env python3
"""
Given a list of P1 barcodes, demultiplex a FASTQ file into many FASTQ files, one for each P1 barcode.
"""

# Author: Ryan Friedman (@rfriedman22)
# Email: ryan.friedman@wustl.edu


import argparse
import logging
import re

import fastq_iter

logger = logging.getLogger(__name__)


def read_sample_map(samples, prefix=""):
    """Reads in a file containing demultiplexing barcodes and prefixes for output FASTQ file, and prepares the files.

    Parameters
    ----------
    samples : str
        Name of file to parse
    prefix : str
        Prefix to give every file

    Returns
    -------
    sample_dict : {str: [file output stream, int, str]}
        Keys are the demultiplexing barcodes, values are lists containing (1) the file handle (2) the number of reads
        that are mapped to that handle (initialized as zero) (3) the prefixes
    """
    sample_dict = {}
    with open(samples) as fin:
        # First value is the P1 barcode, second value is the sample identifier for output
        for line in fin:
            data = line.split()
            barcode = data[0]
            fout = open(f"{prefix}{data[1]}.fastq", "w")
            sample_dict[barcode] = [fout, 0, data[1]]

    return sample_dict


def close_files(sample_dict):
    """Close all the file handles.

    Parameters
    ----------
    sample_dict : dict, {str: [file output stream, int, str]
        A dictionary with lists as values, with the first element being file handles

    Returns
    -------
    None
    """
    for fout, _, _ in sample_dict.values():
        fout.close()


def main(fastq, sample_map, middle, barcode_length, prefix=""):
    sample_dict = read_sample_map(sample_map, prefix=prefix)
    total_reads = 0

    # Prepare the regex for searching the fastq
    # The first part of the match can be any of the demultiplexing barcodes
    pattern = "(" + "|".join(sample_dict.keys()) + ")"
    # The middle sequence must be immediately after the barcode
    pattern += middle
    # And then there must be at least barcode_length number of bases
    pattern += "[ACGTN]{" + str(barcode_length) + ",}"
    pattern = re.compile(pattern)

    # Search for the demux barcodes in each entry of the FASTQ
    for seq_id, seq, qual_id, qual in fastq_iter.fastq_iter(fastq):
        match = pattern.search(seq)
        if match:
            # Get the demultiplexing barcode that matched
            barcode = match.groups(1)[0]
            fout, _, _ = sample_dict[barcode]
            sample_dict[barcode][1] += 1
            fastq_iter.write_fastq(fout, seq_id, seq, qual_id, qual)

        total_reads += 1
        if total_reads % 1000000 == 0:
            logger.info(f"Demultiplexed {total_reads / 1000000} million reads")

    matched_reads = 0
    for _, counts, prefix in sample_dict.values():
        logger.info(f"{counts} reads matched to {prefix} ({counts / total_reads * 100: .2f}%)")
        matched_reads += counts

    logger.info(f"{total_reads - matched_reads} reads did not match to any sample.")
    logger.info(f"{total_reads} reads were sequenced.")

    # Close all the files
    close_files(sample_dict)


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
        "fastq", help="Name of FASTQ file to demultiplex"
    )
    parser.add_argument(
        "sample_map", metavar="demultiplexing_barcodes.txt",
        help="A tab-delimited file that maps each P1 barcode/demultiplex index to a sample identifier. The first "
             "field is a demultiplex index and the second field is the prefix to use for a sample name. For example, "
             "a line might look like the following:\nGCTCGAT\tsample1\nAll reads containing GCTCGAT will be output to "
             "a file called sample1.fastq"
    )
    parser.add_argument(
        "middle", type=str, help="The middle sequence between multiplexing barcode and CRS barcode."
    )
    parser.add_argument(
        "barcode_length", type=int, help="The length of barcodes used to identify CRS."
    )
    parser.add_argument(
        "--prefix", type=str, default="", help="The global prefix to give to all output FASTQ files"
    )
    args = parser.parse_args()

    # Pass args to main
    main(args.fastq, args.sample_map, args.middle, args.barcode_length, prefix=args.prefix)

#!/usr/bin/env python3
"""
Functions for iterating through a FASTQ file.
"""
# Author: Ryan Friedman
# Email: ryan.friedman@wustl.edu

import gzip
import re


def open_file(fastq):
    """Open the FASTQ file, gzip or not, and return the handle."""
    if fastq[-3:] == ".gz":
        return gzip.open(fastq)
    else:
        return open(fastq)


def write_fastq(fout, *lines):
    """Write some lines to the file output stream."""
    for line in lines:
        if type(line) is bytes:
            line = line.decode()
        fout.write(line)


def fastq_iter(fastq):
    """A generator function to parse through one entry in a FASTQ file. This assumes the FASTQ file is Illumina
    format, so each read consists of four lines.

    Parameters
    ----------
    fastq : str
        Name of the file to iterate

    Yields
    -------
    seq_id : str
        Sequence identifier from the sequencer
    seq : str
        The read
    qual_id : str
        Quality score identifier
    qual : str
        Quality scores
    """
    # First check and make sure the file is actually a FASTQ
    with open_file(fastq) as fin:
        seq_id = fin.readline()
        seq = fin.readline().strip()
        qual_id = fin.readline().strip()
        qual = fin.readline().strip()

        # Determine if the string is bytes or characters, if it's bytes than the strings need to be decoded
        decoding = (type(seq_id) == bytes)
        if decoding:
            seq_id = seq_id.decode("utf-8")
            seq = seq.decode("utf-8")
            qual_id = qual_id.decode("utf-8")
            qual = qual.decode("utf-8")

        invalid_seq_id = (seq_id[0] != "@")
        bad_chars = re.compile("[^ACGTN]")
        invalid_seq = (len(bad_chars.findall(seq)) > 0)
        invalid_qual_id = (qual_id != "+")

        # To determine if the quality score is malformed, get the ASCII value of each char and make sure it is
        # between 33 and 75
        invalid_qual = False
        for q in qual:
            ascii = ord(q)
            if ascii < 33 or ascii > 75:
                invalid_qual = True
                break

        if invalid_seq_id or invalid_seq or invalid_qual or invalid_qual_id:
            raise Exception("File is not in FASTQ format.")

    # Iterate through the file
    with open_file(fastq) as fin:
        for seq_id in fin:
            seq = fin.readline()
            qual_id = fin.readline()
            qual = fin.readline()
            if decoding:
                seq_id = seq_id.decode("utf-8")
                seq = seq.decode("utf-8")
                qual_id = qual_id.decode("utf-8")
                qual = qual.decode("utf-8")

            yield (seq_id, seq, qual_id, qual)

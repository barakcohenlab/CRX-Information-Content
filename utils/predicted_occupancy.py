"""
Author: Ryan Friedman (@rfriedman22)
Email: ryan.friedman@wustl.edu
"""
import numpy as np
import pandas as pd
from scipy.special import gamma

import fasta_seq_parse_manip


def peek(fin):
    """ Peek at the next line in a file.

    Parameters
    ----------
    fin : file input stream

    Returns
    -------
    line : str
    """
    pos = fin.tell()
    line = fin.readline()
    fin.seek(pos)
    return line


def gobble(fin):
    """Gobble up lines in the file until we have reached the start of a motif or EOF.

    Parameters
    ----------
    fin : file input stream

    Returns
    -------
    lines : str
        The lines that got gobbled, including newline characters.
    """
    lines = ""
    while True:
        line = peek(fin)
        if len(line) == 0 or line[:5] == "MOTIF":
            break
        else:
            lines += fin.readline()

    return lines


def read_pwm_files(filename):
    """Given a MEME file, read in all PWMs. PWMs are stored as DataFrames, and the list of PWMs is represented as a
    Series, where keys are primary motif identifiers and values are the DataFrames.

    Parameters
    ----------
    filename : str
        Name of the file to read in.

    Returns
    -------
    pwm_ser : pd.Series
        The list of PWMs parsed from the file.
    """
    pwm_ser = {}
    with open(filename) as fin:
        # Lines before the first motif is encountered
        gobble(fin)

        # Do-while like behavior to read in the data
        # Do <read in motif> while not EOF
        while True:
            # MOTIF <motif name> [alternate ID]
            motif_id = fin.readline().split()[1]

            # Empty line
            fin.readline()
            # "letter-probability matrix: [other info]"
            fin.readline()

            # Every line that starts with a space is a new position in the PWM, if the first character is not a space
            # it is not part of the PWM.
            pwm = []
            while peek(fin)[0] == " ":
                pwm.append(fin.readline().split())

            # Make a DataFrame and add to the list
            pwm = pd.DataFrame(pwm, dtype=float, columns=["A", "C", "G", "T"])
            pwm_ser[motif_id] = pwm
            # Read up any extra info such as the URL
            gobble(fin)

            # Check if EOF
            if len(peek(fin)) == 0:
                break

    pwm_ser = pd.Series(pwm_ser)
    return pwm_ser


def ewm_from_letter_prob(pwm_df, pseudocount=0.0001, rt=2.5):
    """Compute an energy weight matrix from a letter probability matrix. Normalize the PWM to the maximum letter
    probability at each position and then compute relative free energies using the formula ddG = -RT ln(p_b,i / p_c,
    i), where p_b,i is the probability of base b, p_c,i is the probability of the consensus base, and ddG is relative
    free energy.

    Parameters
    ----------
    pwm_df : pd.DataFrame
        The letter probability matrix, where each row is a position of the motif and columns represent A, C, G, T.
    pseudocount : float
        Pseudocount value to add to every value to account for zeros in the PWM.
    rt : float
        The value of RT to use in the formula in kJ/mol.

    Returns
    -------
    ewm_df : pd.DataFrame
        The weight matrix of free energies relatives to the consensus sequence.
    """
    pwm_df = pwm_df.copy()
    pwm_df += pseudocount
    # Normalize each position by the most frequent letter to get relative Kd
    pwm_df = pwm_df.apply(lambda x: x / x.max(), axis=1)
    # Convert to EWM
    ewm_df = -rt * np.log(pwm_df)
    ewm_df.columns = ["A", "C", "G", "T"]
    return ewm_df


def ewm_to_dict(ewm):
    """Convert a DataFrame representation of an EWM to a dictionary for faster indexing.

    Parameters
    ----------
    ewm : pd.DataFrame

    Returns
    -------
    ewm_dict : {int: {str: float}}
        Dictionary of dictionaries, where the outer keys are positions, the inner keys are letters, and the values
        are values of the matrix
    """
    ewm_dict = ewm.to_dict(orient="index")
    return ewm_dict


def read_pwm_to_ewm(filename, pseudocount=0.0001, rt=2.5):
    """Read in a file of letter probability matrices, convert them to EWMs, and then convert the DataFrames to
    dictionaries for faster indexting.

    Parameters
    ----------
    filename : str
        Name of the file to read in.
     pseudocount : float
        Pseudocount value to add to every value to account for zeros in the PWM.
    rt : float
        The value of RT to use in the formula in kJ/mol.

    Returns
    -------
    ewm_dict : {int: {str: float}}
        Dictionary of dictionaries, where the outer keys are positions, the inner keys are letters, and the values
        are values of the matrix
    """
    # Wrapper function handle to convert each PWM to an EWM
    converter = lambda x: ewm_from_letter_prob(x, pseudocount, rt)
    # Read in the file
    pwms = read_pwm_files(filename)
    # Convert to EWM dicts
    ewms = pwms.apply(converter).apply(ewm_to_dict)
    return ewms


def energy_landscape(seq, ewm):
    """Scans both strands of a sequence with energy matrix

    Parameters
    ----------
    seq : str
        The sequence to scan.
    ewm : dict {int: {str: float}}
        Dictionary of dictionaries, where the outer keys are positions, the inner keys are letters, and the values
        are delta delta G relative to the consensus sequence.

    Returns
    -------
    fscores, rscores: np.array, dtype=float
        Represents the EWM scores for each subsequence on the forward and reverse strand.
    """
    motif_len = len(ewm.keys())
    # Number of positions where the motif can be scored
    n_scores = len(seq) - motif_len + 1
    fscores = np.zeros(n_scores)
    # Reverse compliment scores
    rscores = fscores.copy()
    r_seq = fasta_seq_parse_manip.rev_comp(seq)

    # Calculate occ for forward and reverse k-mer at every position
    for pos in range(n_scores):
        f_kmer = seq[pos:pos + motif_len]
        r_kmer = r_seq[pos:pos + motif_len]

        # Initialize energy score
        fscore = 0
        rscore = 0
        
        # This is faster than using the enumerate function
        # Calculate the EWM score for the k-mer starting at pos
        for i in range(motif_len):
            fscore += ewm[i][f_kmer[i]]
            rscore += ewm[i][r_kmer[i]]

        fscores[pos] = fscore
        rscores[pos] = rscore

    # rscores needs to be reversed so the indexing corresponds to the appropriate position in the original sequence (
    # i.e. just the compliment, not the reverse compliment)
    rscores = rscores[::-1]

    return fscores, rscores


def occupancy_landscape(seq, ewm, mu):
    """Compute the occupancy landscape by scanning sequence with the energy matrix and then calculate the relative
    free energy for each k-mer subsequence on the forward and reverse strand at chemical potential mu.

    Parameters
    ----------
    seq : str
        The sequence to scan.
    ewm : dict {int: {str: float}}
        Dictionary of dictionaries, where the outer keys are positions, the inner keys are letters, and the values
        are delta delta G relative to the consensus sequence.
    mu : int
        Chemical potential of the TF

    Returns
    -------
    fscores, rscores: np.array, dtype=float
        Represents the occupancy scores for each subsequence on the forward and reverse strand.
    """
    fscores, rscores = energy_landscape(seq, ewm)
    # Convert EWM scores to occupancies
    fscores = 1 / (1 + np.exp(fscores - mu))
    rscores = 1 / (1 + np.exp(rscores - mu))
    return fscores, rscores


def total_landscape(seq, ewms, mu):
    """Compute the occupancy landscape for each TF and join it all together into a DataFrame. Pad the ends of the
    positional information so every TF occupancy landscape is the same length.

    Parameters
    ----------
    seq : str
        The DNA sequence.
    ewms : pd.Series or dict {str: {int: {str: float}}}
        Keys/index are TF names and values are dictionary representations of the EWMs.
    mu : int or float
        TF chemical potential.

    Returns
    -------
    landscape : pd.DataFrame, dtype=float
        The occupancy of each TF at each position in each orientation. Rows are positions, columns are TFs and
        orientations, values indicate the predicted occupancy starting at the position.
    """
    landscape = {}
    seq_len = len(seq)
    # For each TF
    for name, ewm in ewms.items():
        # Get the predicted occupancy and add it to the list
        fscores, rscores = occupancy_landscape(seq, ewm, mu)
        landscape[f"{name}_F"] = fscores
        landscape[f"{name}_R"] = rscores

    # Pad the ends of the lists to the length of the sequence
    for key, val in landscape.items():
        amount_to_add = seq_len - len(val)
        landscape[key] = np.pad(val, (0, amount_to_add), mode="constant", constant_values=0)

    landscape = pd.DataFrame(landscape)
    return landscape


def total_occupancy(seq, ewms, mu):
    """For each TF, calculate its predicted occupancy over the sequence given the energy matrix and chemical
    potential. Then, summarize the information as the total occupancy of each TF over the entire sequence.

    Parameters
    ----------
    seq : str
        The DNA sequence.
    ewms : pd.Series or dict {str: {int: {str: float}}}
        Keys/index are TF names and values are dictionary representations of the EWMs.
    mu : int or float
        TF chemical potential.

    Returns
    -------
    occ_profile : pd.Series, dtype=float
        The total occupancy profile of each TF on the sequence.
    """
    occ_landscape = total_landscape(seq, ewms, mu)
    occ_profile = {}
    # Add together F and R strand
    if type(ewms) is dict:
        keys = ewms.keys()
    else:
        keys = ewms.index
    for tf in keys:
        occ_profile[tf] = occ_landscape[[f"{tf}_F", f"{tf}_R"]].sum().sum()

    occ_profile = pd.Series(occ_profile)
    return occ_profile


def all_seq_total_occupancy(seq_ser, ewm_ser, mu, convert_ewm=True):
    """Calculate the total predicted occupancy of each TF over each sequence.

    Parameters
    ----------
    seq_ser : pd.Series, dtype=str
        Representation of FASTA file, where each value is a different sequence. Index is the FASTA header.
    ewm_ser : pd.Series, dtype=pd.DataFrame
        Each value of the series is an energy matrix for a different TF.
    mu : int or float
        TF chemical potential.
    convert_ewm : bool
        If True, convert the EWMs from a DataFrame to a dictionary representation. If False, assumes that the EWMs
        have already been converted.

    Returns
    -------
    occ_df : pd.DataFrame, shape=[n_seq, n_tf]
        Total predicted occupancy of each TF over each sequence. Rows are sequences with same index as seq_ser,
        columns represent different TFs.
    """
    # Convert the EWMs to dictionary representations for speedups
    if convert_ewm:
        ewm_ser = {name: ewm_to_dict(ewm) for name, ewm in ewm_ser.iteritems()}

    seq_ser = seq_ser.str.upper()
    occ_df = seq_ser.apply(lambda x: total_occupancy(x, ewm_ser, mu))

    return occ_df


def boltzmann_entropy(occupancies, diversity_cutoff=0.5, log=np.log2):
    """Given a list of TF occupancies, compute total occupancy, diversity, and Boltzmann entropy.

    Parameters
    ----------
    occupancies : pd.Series
        Predicted occupancy for a collection of TFs on a given sequence.
    diversity_cutoff : float
        Cutoff to call a TF "occupied" on the sequence.
    log : Function handle
        Function to use for computing the log. Default is log2 so entropy is in bits, natural log should be used for
        biophysical applications.

    Returns
    -------
    result : pd.Series
        The total occupancy, diversity, and entropy of the provided sequence.
    """
    # Calculate total occupancy of all TFs on the sequence
    total_occ = occupancies.sum()
    # Count how many of the TFs are occupied, i.e. have motifs present in the sequence
    diversity = (occupancies > diversity_cutoff).sum()
    # Since the occupancies are continuous values, we need to use the Gamma function to compute entropy. Gamma(n+1)=n!
    # W = N! / prod(N_i!)
    microstates = gamma(total_occ + 1) / (occupancies + 1).apply(gamma).product()
    # S = log W
    entropy = log(microstates)

    result = pd.Series({
        "total_occupancy": total_occ,
        "diversity": diversity,
        "entropy": entropy
    })
    return result

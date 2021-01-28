#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -o dreme-%A.out
#SBATCH -e dreme-%A.err
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL

ml meme/5.0.4

# A wrapper for running DREME-py3.
if [ "$#" -ne 6 ]
then
    echo "Usage: dreme.sh positives.fasta negatives.fasta outDir/ mink maxk motifDatabase.meme"
    echo "\tmink is the minimum motif size to search for and maxk is the maximum motif size to search for."
    exit 1
fi

positives=$1
negatives=$2
outdir=$3
mink=$4
maxk=$5
database=$6

dreme-py3 -oc "${outdir}" -p "${positives}" -n "${negatives}" -dna -eps -mink "${mink}" -maxk "${maxk}" -e 0.05
cd "${outdir}"
sh ~/CRX-Genomic-Analysis/utils/tomtom.sh dreme.html "${database}" .

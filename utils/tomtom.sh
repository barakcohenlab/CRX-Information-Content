#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -o tomtom-%A.out
#SBATCH -e tomtom-%A.err
#SBATCH --mail-type=END,FAIL

ml meme/5.0.4

# A simple wrapper to run TOMTOM

if [ "$#" -ne 3 ]
then
    echo "Usage: tomtom.sh foundMotifs.meme database.meme outDir/"
    exit 1
fi

foundMotifs=$1
database=$2
outdir=$3

tomtom -oc "${outdir}" "${foundMotifs}" "${database}"

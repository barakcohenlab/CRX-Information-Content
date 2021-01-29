#!/bin/bash
sbatch --array=1-2 ~/CRX-Information-Content/utils/demultiplexAndBarcodeCount.sh fastqToSequence.txt CATGC 9 ~/CRX-Information-Content/utils/

#!/bin/bash
sbatch --array=1-2 ~/CRX-Information-Content/utils/demultiplexAndBarcodeCount.sh fastqToSample.txt CATGC 9 ~/CRX-Information-Content/utils/

# CRX Information Content
This repository contains all code, reasonably sized data, and directory structures necessary to reproduce figures in "Information Content Differentiates Enhancers From Silencers in Mouse Photoreceptors" by Friedman et al., 2021.



## Prerequisites
All necessary software packages are described in `bclab.yml` with the versions used to process data and produce figures. The easiest way to install these packages is by downloading  [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and then typing the command `conda create -f bclab.yml`. You can then activate the virtual environment using `conda activate bclab`. Then, from this directory, launch Jupyter with the command `jupyter lab`.

In addition, you will need to install gkmSVM. Download the C++ source code from the [Beer lab website](http://www.beerlab.org/gkmsvm/) and compile the code following the README. Be sure to change `GKMSVM_PATH` in `utils/gkmsvm.py` to point to wherever the binaries were installed. To generate motifs from k-mer scores, you may need to convert the script `svmw_emalign.py` from Python 2 to Python 3. You can do this with the command `2to3`.

All motif analysis was performed using version 5.0.4 of the MEME suite on the WashU High Throughput Computing Facility ([HTCF](http://htcf.wustl.edu/)). There are wrapper scripts to run DREME and TOMTOM on HTCF using [SLURM](https://slurm.schedmd.com/documentation.html) located in `utils`.

## Organization of the repository
* All Jupyter notebooks used to process data and genreate figures are in the base directory. They should be run in the numbered order.
* `dataset_names.txt` contains a mapping from file names in the `Data` directory to the corresponding dataset number in the manuscript.
* `Data` contains FASTA files of the libraries as well as a BED file with the mm10 coordinates of the sequences. Some of the notebooks will generate additional files.
    * `ActivityBins` is where FASTA files will be generated for different subsets of the data. This directory also contains the results from running DREME and TOMTOM.
    * `Downloaded` contains all datasets that were downloaded for analysis. `ChIP` contains ChIP-seq peaks for [MEF2D](http://dx.doi.org/10.1016/j.neuron.2015.02.038) and [NRL](http://dx.doi.org/10.1371/journal.pgen.1002649) in mm10 coordinates, `CrxMpraLibraries` contains data from [White et al., *PNAS*, 2013](http://dx.doi.org/10.1073/pnas.1307449110), and `Pwm` contains the 8 TFs used for the motif analysis. The [full mouse HOCOMOCO database, v11](https://hocomoco11.autosome.ru/) should also be downloaded to this directory.
    * `Polylinker` and `Rhodopsin` contain the raw barcode counts from the respective experiments. Note that these are the same files that are submitted to GEO, but with slightly different file names because GEO does not have the directory structure used here. Notebooks 1 through 3 generate additional files in these directories.
* `Figures` will contain all figures and supplemental figures for the manuscript after running the notebooks. The only figure not generated with Jupyter notebooks is Supplementary Figure 4, which shows the results from the motif analysis. However, the TOMTOM results are saved in `Data/ActivityBins`.
* `Models` will be generated in notebook 6. It contains different SVM and logistic regression models.
* `Sequencing` contains supplementary files and scripts to demultiplex the FASTQ files and count barcodes (see below).
* `utils` contains files with Python functions used to process, analyze, and visualize the data; wrapper scripts for running DREME and TOMTOM on SLURM; and scripts to demultiplex FASTQ files and count barcodes.

## Demultiplexing FASTQ files and counting barcodes
This task must be performed on a cluster such as HTCF. The `run.sh` scripts in the `Polylinker` and `Rho` directories submit SLURM jobs. In order to complete this task, clone the repository into your home directory on HTCF. Then, copy all files from the `Polylinker` or `Rho` directory into your `scratch` space. Download the FASTQ file from SRA to the same directory in your `scratch` space. Finally, type `sh run.sh` to submit the job to SLURM. If you want to get an email when the script is done running, add `--mail-user=<your email here>` immediately following `sbatch` in the script.

Once the job has completed, there will be `.counts` files in your `scratch` space. There will also be some logs you can check and histograms showing the distribution of barcode counts in each sample. You can now copy the `.counts` files into your `Data` directory to use on your local machine. These are the same files already in the repository and submitted to GEO.

## Brief description of Jupyter notebooks
1. Processes library 1 with the *Rho* promoter from raw barcode counts to activity scores and performs statistics comparing each sequence to the basal *Rho* promoter alone.
2. Same as notebook 1, but processes library 2 with the *Rho* promoter.
3. Processes library 1 and 2 with the Polylinker from raw barcode counts to activity scores.
4. Joins together the data generated from the first 3 notebooks into one large table and annotates sequences for their activity groups and overlap with TF ChIP-seq datasets. Generates supplementary figures 1 and 2.
5. Computes predicted occupancy for each TF on each sequence, then computes information content and related metrics.
6. Fits different models on the data with cross-validation and performs additional validation on the TF occupancy logistic regression model.
7. Generates all other figures for the manuscript, performs statistics, and displays additional data that might be useful.
8. Generates logos for additional PWMs used in this study.

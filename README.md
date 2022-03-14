# Language Modeling of Viral Glycoproteins- Favour Obuseh and Siddharth Iyer
Practice repository setup for BE 20.440 class. 
This repository contains code to analyze viral glycoproteins with language modeling. The methods are based on those described in Hie et al. 2021.

## Overview
By running rhabdo.py and rhabdo_embed.py, the code will train an LSTM on all rhabdoviridae sequences and then embed all non-rabies virus glycoproteins. Then it produces a Principal component analysis plot (PCA) colored by strain of the non-rabies viruses.

## Data
The data was generated by searching for glycoproteins on VIPR (https://www.viprbrc.org/brc/home.spg?decorator=vipr) within all Rhabdoviridae genomes.

## Folder Structure
The rhabdoviridae sequences are in the data folder. Figures and a trained model are in the figures folder. All necessary codes/ modules for our project are in the main folder. There is a code to produce figure folder which contains an IPNYB file that can be copied.

## Installation 
Clone this repository and create a virtual environment with python 3.8 with the packages in requirements.txt



# ML_MB_Norway

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15021796.svg)](https://doi.org/10.5281/zenodo.15021796)

## Overview
Machine learning for glacier mass balance modeling in Norway.

Repository for manuscript: **Machine learning improves seasonal mass balance prediction for unmonitored glaciers** by Sjursen et al. (to be submitted to The Cryosphere).

This repository is a prototype of the MassBalanceMachine. We recommend users interested in applying and developing machine learning for glacier mass balance modelling to check out the MassBalanceMachine repository for the most recent version currently under Development: https://github.com/ODINN-SciML/MassBalanceMachine

## Repository Structure
- **`src/`**: Contains Notebooks for preprocessing, ML pipeline and analysis of results/figures.
- **`src/Data`**: Data files of point mass dataset and preprocessing steps. Data files for model comparison.
- **`src/Training_cluster`**: Files for trained model. 

## Conda environments
- **`environment.yml`**: Environment required to run most Notebooks.
- **`oggm_recommended_env.yml`**: Environment for running data retreival with the OGGM pipeline (not available for Windows).
- **`skyview_env.yml`**: Environment for computing skyview factor using the `rvt` package (requires Python 3.7)  

## Data
Dataset of point mass balance for glaciers in Norway used data used in this study can be found at:
https://doi.org/10.58059/sjse-6w92

## License
This project is licensed under the terms of the MIT License. See LICENSE for details.

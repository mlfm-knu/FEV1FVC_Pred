# Integrated Multi-features with Tabular prior-data fitted network differentiate chronic obstructive pulmonary disease patients

## 1. Introduction

This tool was developed by the Machine Learning and Fluid Mechanics Lab at Kyungpook National University. It aims to support the research described in the article titled "Integrated Multi-features with Tabular prior-data fitted network differentiate chronic obstructive pulmonary disease patients".

## 2. License

> This software is provided under the Machine Learning and Fluid Mechanics Lab License. 
> You are free to use, modify, and distribute this software for research purposes, provided that you cite the original paper (see section 3) and acknowledge the developers. Commercial use requires explicit permission from the authors.


## 3. Reference Paper

If you use this tool in your research, please cite the following paper:

> [**Integrated Multi-features with Tabular prior-data fitted network differentiate chronic obstructive pulmonary disease patients**]

Chau N-K, Kim WJ, Lee CH, Chae KJ, Jin GY, Choi S, Integrated Multi-features with Tabular prior-data fitted network differentiate chronic obstructive pulmonary disease patients.


## 4. Installation

**Prerequisites:**

* All necessary Python packages and their versions are listed in the requirements.txt file


**Installation Steps:**

1.  Clone the repository (if you are distributing the code via a repository):
    ```
    git clone [REPOSITORY_URL]
    ```

2. Install the required Python libraries. It is highly recommended to create a virtual environment before installing the dependencies to avoid conflicts with other projects. You can use `conda` or `venv` for this purpose.


## 5. Usage

### Data Preparation

### Data Structure

The project expects a specific folder and file structure under the `data/` directory. If you are using your own data, please organize it in the following manner:

data/
├── airway_trees/
│   └── airway_1.xlsx
│   └── airway_2.xlsx	
│   └── ...
├── PRM_images/
│   └── PRM_1.nii
│   └── PRM_2.nii
│   └── ...
└── data.xlsx

* `airway_trees/`: This folder contains detailed information on airway trees. For more details, please refer to our article and the example file provided.

* `PRM_images/`: This folder contains Parametric Response Mapping (PRM) images that have been resampled to a uniform size of 64x64x64.

* `data.xlsx`: This file contains demographic and Quantitative Computed Tomography (QCT) imaging data. Refer to our article and the example file for more specifics.


The target endpoint is 0 for non-COPD subjects and 1 for COPD patients.  For a more detailed explanation of the training features, please refer to our article.

We also provide a pre-trained scaler and pre-trained models in the `models` folder.



### Running the Scripts

Run the below command,
    ```
    python Run_script.py
	```
	
The results of the script will be saved in the `output` folder.
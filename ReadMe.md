# Integrated Multi-features with Tabular prior-data fitted network differentiate chronic obstructive pulmonary disease patients

## 1. Introduction

This tool was developed by the Machine Learning and Fluid Mechanics Lab at Kyungpook National University. It aims to support the research described in the article titled "Integrated Multi-features with Tabular prior-data fitted network differentiate chronic obstructive pulmonary disease patients".

## 2. License

> This software is provided under the Machine Learning and Fluid Mechanics Lab License. 
> You are free to use, modify, and distribute this software for research purposes, provided that you cite the original paper (see section 3) and acknowledge the developers. Commercial use requires explicit permission from the authors.

Apache License Version 2.0, January 2004 http://www.apache.org/licenses/

Additional Information
This program is registered software (Copyright Registration No. C-XXX, Korea Copyright Commission).
Certain parts of the algorithm are protected under Patent No. XX (KIPO).
Copyright (c) 2024 MLFM Lab, Kyungpook National University.
Some codes in this repository are modified from Dual deep mesh prior.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

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
You need to request trained models to MLFM laboratory because of the file size (e-mail: s-choi@knu.ac.kr)

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

Apache License Version 2.0, January 2004 http://www.apache.org/licenses/

Additional Information
This program is registered software (Copyright Registration No. C-2023-026549, Korea Copyright Commission).
Certain parts of the algorithm are protected under Patent No. 10-2759618 (KIPO).
Copyright (c) 2024 MLFM Lab, Kyungpook National University.
Some codes in this repository are modified from Dual deep mesh prior.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

### Running the Scripts

Run the below command,
    ```
    python Run_script.py
	```
	
The results of the script will be saved in the `output` folder.

# Cross-geometry transfer learning for neural operators via spectral alignment and adaptive fine-tuning
## Introduction
We present a spectral alignment framework that enables efficient cross-geometry transfer learning for neural operators. First, we construct a shared spectral representation by computing aligned spectral bases for source and target geometries via approximate joint diagonalisation. These aligned spectral bases are integrated with a frequency-domain neural operator to train a source model for source geometry. To adapt this model to the target geometry with very limited data, we introduce a dedicated two-stage fine-tuning strategy: spectral alignment is first refined, followed by fine-tuning model parameters.

![image](https://github.com/code-cl/TLNO_via_Spectral_Alignment/blob/main/TLNO_Framework.png)
## Repository Structure
Each case folder follows the same layout and contains the following subdirectories:
```text
<case_name>/
├── Data/
├── DeepONet/
│   ├── source_model.py
│   └── target_model.py
├── Get_LBO/
│   ├── 1_Calculate_lbo.py
│   └── 2_Save_indicator_function.py
└── Iterative_NOs/
    ├── main_train.py
    ├── main_transfer.py
    └── main_transfer_cbf.py
```
## Usage
For all cases, you can directly run the codes by executing the '.py' files to quickly obtain the results. Note that each experiment is repeated three times, the same setup as in our paper. Each case also retains the setting of hyperparameters in the paper.

`Data/`
Stores the training data and the mesh information of part geometry used by the case.

`DeepONet/`
Contains the implementation of the TL-DeepONet method.
- `source_model.py` — trains the source model.
- `target_model.py` — fine-tunes the source model with a small amount of target data to obtain the target model.

`Get_LBO/`
Contains the code for computing the Laplace–Beltrami operator (LBO) basis and the cross-geometry correspondence function. `Data/` already contains the calculated LBO base files.
- `1_Calculate_lbo.py` — computes the LBO from the given mesh information.
- `2_Save_indicator_function.py` — computes and saves the cross-geometry correspondence function.

`Iterative_NOs/`
Contains the implementation of Transolver, NORM, and NORM++.
- `main_train.py` — trains the source model. The model type (Transolver / NORM / NORM++) is selectable.
- `main_transfer.py` — fine-tunes the model for transfer learning. The model type (Transolver / NORM / NORM++) is selectable.
- `main_transfer_cbf.py` — implements the TL(CBF)-NORM++ method.

## Data Preparation
All datasets are provided in the `Data/` folder of each corresponding case. The data sources for each specific case are described below:

`Darcy`
The pressure fields under different permeability coefficient fields are obtained by solving the Darcy flow equation using MATLAB's **SOLVEPDE** toolbox. The data acquisition process follows the literature: *"Lu, L., Meng, X., Cai, S., Mao, Z., Goswami, S., Zhang, Z., & Karniadakis, G. E. (2022). A comprehensive and fair comparison of two neural operators (with practical extensions) based on fair data. Computer Methods in Applied Mechanics and Engineering, 393, 114778"*.

`FlowBench`
The dataset is sourced from the **FlowBench** dataset introduced  in *"Tali, R., Rabeh, A., Yang, C. H., Shadkhah, M., Karki, S., Upadhyaya, A., ... & Ganapathysubramanian, B. (2024). Flowbench: A large scale benchmark for flow simulation over complex geometries. arXiv preprint arXiv:2409.18032"*.

`Composite case`
The deformation fields under different temperature fields are obtained via **COMSOL**. The simulation process and parameter settings reference: *"Liu, S., Li, Y., Gan, J., Yang, Z., Zhou, J., & Hao, X. (2023). Active control of cure-induced distortion for composite parts using multi-zoned self-resistance electric heating method. Journal of Manufacturing Processes, 93, 47-59"*.

`Soft material`
The response shapes under different material distributions are obtained via **ABAQUS**. The simulation process and parameter settings reference: *"Sun, X., Yue, L., Yu, L., Forte, C. T., Armstrong, C. D., Zhou, K., ... & Qi, H. J. (2024). Machine learning-enabled forward prediction and inverse design of 4D-printed active plates. Nature Communications, 15(1), 5509"*.

## Acknowledge
We appreciate the following GitHub repos a lot for their valuable code:  
https://github.com/katiana22/TL-DeepONet  
https://github.com/thuml/Transolver  
https://github.com/gengxiangc/NORM  
https://github.com/Deep-MI/LaPy  


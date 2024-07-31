# Evaluation of deep learning approaches for drug combination synergy prediction in the context of cancer cell lines based on TDC benchmark

## Overview

This repository contains scripts to train models for the tasks provided in Drug Combination Benchmark Group from the [Therapeutics Data Commons (TDC)](https://tdcommons.ai/). The tasks are related to prediction of different drug combination synergy scores viz. CSS, BLISS, HSA, Loewe and ZIP in context of different cancer cell lines. 

More details about the tasks can be found [here](https://tdcommons.ai/benchmark/drugcombo_group/overview/). 

## Description of tasks

The predictive models for the tasks accept following three inputs:
1. Drug 1 SMILES
2. Drug 2 SMILES
3. Cell line context

The output of each model is the corresponding synergy score (from CSS, BLISS, HSA, Loewe and ZIP). 
All five tasks are regression tasks.

## Approaches

The code in this repository implements two different approaches for the five tasks.

### Approach 1: CongFu layers based approach

This approach is based on model architecture described in the paper by Tsepa, Oleksii, et al. 2023.
The original model architecture models the synergy prediction as binary classification problem.
We modify this model architecture for the regression task along with other modifications as described in the report. 

### Approach 2: Pre-trained molecular representation based approach

This approach uses pre-trained molecular language model for obtaining the drug molecule features from their SMILES representation.

Both approaches use cell line feautures available through TDC benchmark dataset.

## Setup
Setup the environment for the project using `tdcdc.yml` by running following command.
```
conda env create -f tdcdc.yml
```
Alternatively, the environment can be setup using following steps:
```
conda create --name tdcdc python==3.9.18
conda activate tdcdc
pip install -r requirements.txt
```

## Usage
The model training pipeline for the five target score using TDC data can be run by executing following commands:

For using Approach 1:
```python
python drugcomb_synergy_tdc.py congfu
```
For using Approach 2:
```python
python drugcomb_synergy_tdc.py molformer
```

The code in `drugcomb_synergy_tdc.py` trains five models for the five target variables using five different data splits according the TDC Drug combination benchmark guidelines. 
The performance metrics are calculated using the `tdc` package utilties and final results are printed.

## Contact
Please contact [Ketan sarode](ketan.sarode@ics.innoplexus.com) if you have any questions!

Feel free to contribute to this project by opening issues or submitting pull requests. For any questions or inquiries, please contact the repository maintainer.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## References

Tsepa, Oleksii, et al. "CongFu: Conditional Graph Fusion for Drug Synergy Prediction." arXiv preprint arXiv:2305.14517 (2023).

Ross, Jerret, et al. "Large-scale chemical language representations capture molecular structure and properties." Nature Machine Intelligence 4.12 (2022): 1256-1264.

## Powered By

<img src="https://www.nvidia.com/en-us/about-nvidia/legal-info/logo-brand-usage/_jcr_content/root/responsivegrid/nv_container_392921705/nv_container/nv_image.coreimg.100.630.png/1703060329053/nvidia-logo-vert.png" alt="NVIDIA" height="100"/>
<img src="https://tdcommons.ai/logonav.png" alt="tdc" height="80"/>
<img src="https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png" alt="Pytorch" height="80"/>
<img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.png" alt="HuggingFace" height="100"/>
<img src="https://lightning.ai/static/media/logo-with-text-dark.bdcfdf86bccb9812ed1d3ec873e9e771.svg" alt="PytorchLightning" height="100"/>
<img src="https://pypi-camo.freetls.fastly.net/085259150ce4425b27cfcb72d8f48df9640b73d3/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f7079672d7465616d2f7079675f737068696e785f7468656d652f6d61737465722f7079675f737068696e785f7468656d652f7374617469632f696d672f7079675f6c6f676f5f746578742e7376673f73616e6974697a653d74727565" alt="PytorchGeometric" height="100"/>

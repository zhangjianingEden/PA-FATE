# PA-FATE
Additional materials for paper "Fast Feature Selection for Structured Data via Progress-Aware Multi-Task Deep Reinforcement Learning" submitted to ICDE 2023.
## :page_facing_up: Description
PA-FATE is a novel progress-aware Multi-Task Deep Reinforcement Learning (MT-DRL) based method for fast feature selection called PA-FATE. It consists of a basic
framework for knowledge generalization and transfer based on MT-DRL, called “FATE”, a dynamic inter-task resource scheduler called “Inter-Task Scheduler” which can dynamically allocate resources to ensure balanced learning over multiple historical tasks, and a tree-based intra-task optimizer called “Intra-Task Explorer” to enable efficient search over large feature space for each task. 
## :wrench: Dependencies
- Python == 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch == 1.8.1](https://pytorch.org/)
- NVIDIA GPU (RTX 3090) + [CUDA 11.1](https://developer.nvidia.com/cuda-downloads)
### Installation
1. Clone repo
    ```bash
    git clone https://github.com/zhangjianingEden/PA-FATE.git
    cd PA-FATE
    ```
2. Install dependent packages
    ```
    pip install -r requirements.txt
    ```
## :zap: Quick Inference

Get the usage information of the project
```bash
cd code
python main.py -h
```
Then the usage information will be shown as following
```
usage: main.py [-h] env_name method_name mode

positional arguments:
  dataset_name     the name of dataset
  method_name  the name of method
  mode         train or test
 
optional arguments:
  -h, --help   show this help message and exit

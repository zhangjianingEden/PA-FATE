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
  dataset_name          the name of dataset
  max_fea_ratio         the maximum ratio of selectable features
  mode                  train or test
  further_train_iter    the No. of iterations for further train on unseen tasks
 
optional arguments:
  -h, --help   show this help message and exit
```
Test the trained models provided in [PA-FATE/results](https://github.com/zhangjianingEden/PA-FATE/tree/main/results).
```
python main.py Emotions 0.5 test
```
## :computer: Training

We provide complete training codes for PA-FATE.<br>
You could adapt it to your own needs.

1. If you don't have NVIDIA RTX 3090, you should comment these two lines in file
[PA-FATE/util.py](https://github.com/zhangjianingEden/PA-FATE/tree/main/util.py).
	```
	[20]  torch.backends.cuda.matmul.allow_tf32 = False
	[21]  torch.backends.cudnn.allow_tf32 = False
	```
2. You can modify the config files 
[PA-FATE/method/conf_temp.py](https://github.com/zhangjianingEden/PA-FATE/tree/main/method/conf_temp.py) for method.<br>
For example, you can control the number of training iteration by modifying this line
	```
	[17]  'train_iter': 6000,
	```
3. Training
	```
	python main.py Emotions 0.5 train
	```
	The results will be stored in [PA-FATE/results/Emotions/max_fea_ratio_0.5](https://github.com/zhangjianingEden/PA-FATE/tree/main/results/Emotions/max_fea_ratio_0.5).
## :checkered_flag: Testing
1. Testing without further training
	```
	python main.py Emotions 0.5 test
	```
2. Testing with further training
	```
	python main.py Emotions 0.5 test 500
	```
## :scroll: Acknowledgement

Corresponding author: Meihui Zhang, Zhaojing Luo.

## :e-mail: Contact

If you have any question, please email `zhangjianingbit@gmail.com`.

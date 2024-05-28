# DoRA: Enhancing Parameter-Efficient Fine-Tuning with Dynamic Rank Distribution

This repository contains the code for the paper "DoRA: Enhancing Parameter-Efficient Fine-Tuning with Dynamic Rank Distribution."

# Quick Start

## 1. Install the requirements
```pip install -r requirements.txt```

## 2. Download datasets and models
```python Src/Download_Data.py```
This will download the datasets and models required to the cache folder, typically in ```~/.cache/huggingface/```.

## 3. Pre-process the data
```python Src/Dataset_Tokenize/Dataset_Tokenize.py```
This will tokenize the datasets and save the pre-processed data in the ```./Data``` folder.

To change the hyperparameters for input sequence length, edit ```Src/Dataset_Tokenize/Tokenize_Config.py```, and change the value in the specific Dataset_Length_Enum.

## 4. Run fine-tuning
Our code searches multiple hyperparameter combinations on multiple GPUs simultaneously. You will need to specify the hyperparameter combinations to be searched before running and specify the GPU usage configuration before running or modify it during execution.

Edit ```Src/Finetune_And_Benchmark/Build_Finetune_Config_List.py``` to specify the hyperparameter combinations to be searched. Check the comments in the file for more details.

Edit ```Src/Finetune_And_Benchmark/launcher_config.json```:
- ```gpu_id_list```: GPUs to run the experiments on,
- ```max_task_per_gpu```: maximum number of tasks to run on each GPU,
- ```min_free_gpu_memory_gb```: minimum remaining GPU memory to avoid OOM. If the remaining GPU memory is less than this value, new tasks will not be launched on this GPU even if the number of running tasks is less than ```max_task_per_gpu```,
- ```max_cpu_memory_usage_percentage```: max CPU memory usage percentage, similar to ```min_free_gpu_memory_gb```.

If ```gpu_id_list``` and ```max_task_per_gpu``` are changed during running, running tasks might exit to fit the new configuration, the checkpoint will be saved, and the task will be relaunched later.

Run ``` python Src/Finetune_And_Benchmark/Launcher.py``` to start the fine-tuning process. The logs will be saved in the ```./Log``` folder. The checkpoints will be saved in the ```./Checkpoint``` folder.

## 5. Summarize the results
```python Src/Finetune_And_Benchmark/Summarize_Results.py```

This will summarize the index of each hyperparameter combination, as well as the correlation between each hyperparameter and the index.

# Citation
If you find this repository useful, please cite our paper:
```bibtex
@misc{mao2024dora,
    title={DoRA: Enhancing Parameter-Efficient Fine-Tuning with Dynamic Rank Distribution},
    author={Yulong Mao and Kaiyu Huang and Changhao Guan and Ganglin Bao and Fengran Mo and Jinan Xu},
    year={2024},
    eprint={2405.17357},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

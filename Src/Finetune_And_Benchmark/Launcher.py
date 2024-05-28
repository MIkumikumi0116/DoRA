import re
import sys
import time
import json
import shutil
import psutil
import subprocess
import multiprocessing
from pathlib import Path

sys.path.append(Path(__file__).parent.as_posix())
from Build_Finetune_Config_List import build_finetune_config_list

sys.path.append((Path(__file__).parent.parent / 'Components').as_posix())
from Utils_And_Config import current_time



class Launcher:
    def launcher_logic(self):
        self.task_list                  = build_finetune_config_list()
        self.free_gpu_id_list           = []
        self.launcher_config_dict       = {} # declaration, set in reload_launcher_config
        # launcher_config_dict: {
        #     min_free_gpu_memory_gb:          only launch task when free gpu memory is larger than this value, will not kill running task
        #     max_task_per_gpu:                at most this number of tasks can be launched on a single gpu,    will not kill running task
        #     max_cpu_memory_usage_percentage: only launch task when cpu memory usage is lower than this value, will kill running task if higher
        #     gpu_id_list:                     only launch task on gpu in this list,                            will kill running task if running on other gpu
        # }


        self.finished_task_index_list   = []
        self.unassigned_task_index_list = [index for index in range(len(self.task_list))]
        self.process_list               = [] # [(task_index, task_process, gpu_id)]

        self.update_free_gpu_id_list()

        while True:
            self.check_running_process_finished()
            self.fit_launcher_config()
            self.dispatch_new_process()

            if len(self.finished_task_index_list) == len(self.task_list):
                break

            if len(self.free_gpu_id_list) == 0:
                time.sleep(30)
                self.update_free_gpu_id_list()

        print(f'{current_time()} all tasks finished')

    def reload_launcher_config(self):
        while True:
            try:
                with open((Path(__file__).parent / 'launcher_config.json').as_posix(), 'r', encoding = 'utf8') as file:
                    self.launcher_config_dict = json.load(file)
            except:
                print(f'{current_time()} load launcher_config.json failed, retry in 3 seconds')
                time.sleep(3)
                continue
            else:
                break

    def update_free_gpu_id_list(self):
        self.reload_launcher_config()

        nvidia_smi     = subprocess.run(['nvidia-smi'], stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True).stdout
        gpu_usage_list = re.findall(r"\|   (\d+)  [^|]+\|[^|]+\|[^|]+\|\n\|[^|]+\|\s+(\d+)MiB / (\d+)MiB", nvidia_smi, re.MULTILINE)
        gpu_usage_list = [(int(gpu_id), int(used_memory), int(total_memory)) for gpu_id, used_memory, total_memory in gpu_usage_list]

        free_gpu_id_list = [
            gpu_id
            for gpu_id, used_memory, total_memory in gpu_usage_list
            if total_memory - used_memory > self.launcher_config_dict['min_free_gpu_memory_gb'] * 1024
        ]
        free_gpu_id_list = [gpu_id for gpu_id in free_gpu_id_list if gpu_id in self.launcher_config_dict['gpu_id_list']]

        task_num_on_each_gpu_dict = {gpu_id: sum(1 for process_info in self.process_list if process_info[2] == gpu_id) for gpu_id in free_gpu_id_list}
        free_gpu_id_list = [gpu_id for gpu_id in free_gpu_id_list if task_num_on_each_gpu_dict[gpu_id] < self.launcher_config_dict['max_task_per_gpu']]

        self.free_gpu_id_list = free_gpu_id_list

    def check_running_process_finished(self):
        process_index = 0
        while process_index < len(self.process_list):
            task_index, task_process, _ = self.process_list[process_index]
            if not task_process.is_alive():
                task_tag = self.task_list[task_index].task_tag
                log_path = self.task_list[task_index].log_path

                if log_path.exists():
                    if (log_path / 'error.txt').exists() and (log_path / 'error.txt').stat().st_size > 0: # error ocurred
                        with open(log_path / 'error.txt', 'r', encoding = 'utf8') as file:
                            print(file.read())
                        task_succeed_flag = False
                    elif not (log_path / 'task_info.json').exists() or not (log_path / 'error.txt').exists() or (log_path / 'uncompleted.txt').exists(): # task not initialized or not finish
                        task_succeed_flag = False
                    else:
                        task_succeed_flag = True
                else:
                    task_succeed_flag = False

                if task_succeed_flag == False:
                    if log_path.exists():
                        shutil.rmtree(log_path)

                    self.unassigned_task_index_list.append(task_index)
                    print(f'{current_time()} task {task_index + 1} {len(self.finished_task_index_list)}/{len(self.task_list)} {task_tag} failed and will be relaunched later')
                else:
                    self.finished_task_index_list.append(task_index)
                    print(f'{current_time()} task {task_index + 1} {len(self.finished_task_index_list)}/{len(self.task_list)} {task_tag} finished')

                self.process_list.pop(process_index)

            else:
                process_index += 1

    def fit_launcher_config(self):
        self.reload_launcher_config()

        process_index = len(self.process_list) - 1 # terminate the most recently initiated process first
        while process_index >= 0:
            task_index, task_process, gpu_id = self.process_list[process_index]

            process_should_be_killed_flag = False
            if psutil.virtual_memory().percent > self.launcher_config_dict['max_cpu_memory_usage_percentage']:
                process_should_be_killed_flag = True
            if gpu_id not in self.launcher_config_dict['gpu_id_list']:
                process_should_be_killed_flag = True

            if process_should_be_killed_flag:
                task_process.kill()
                time.sleep(5) # wait for releasing cpu memory and closing handle in task_process

                task_tag = self.task_list[task_index].task_tag

                self.unassigned_task_index_list.append(task_index)
                print(f'{current_time()} task {task_index + 1} {len(self.finished_task_index_list)}/{len(self.task_list)} {task_tag} killed for exceeding the resource limit and will be relaunched later')

                self.process_list.pop(process_index)

            process_index -= 1

    def dispatch_new_process(self):
        if (
            len(self.unassigned_task_index_list) == 0 or
            psutil.virtual_memory().percent > self.launcher_config_dict['max_cpu_memory_usage_percentage'] or
            len(self.free_gpu_id_list) == 0
        ):
            return

        task_index = self.unassigned_task_index_list.pop(0)
        task_tag   = self.task_list[task_index].task_tag

        task_finished_in_previous_launch_flag = self.check_task_finished_in_previous_launch(task_index)
        if task_finished_in_previous_launch_flag == True:
            self.finished_task_index_list.append(task_index)
            print(f'{current_time()} task {task_index + 1} {len(self.finished_task_index_list)}/{len(self.task_list)} {task_tag} finished')
            return

        gpu_id  = self.free_gpu_id_list.pop(0)
        task_process = multiprocessing.Process(target = self.launch_task_process, args = (task_index, gpu_id))
        task_process.start()
        self.process_list.append((task_index, task_process, gpu_id))

        print(f'{current_time()} task {task_index + 1} {len(self.finished_task_index_list)}/{len(self.task_list)} {task_tag} launching on gpu {gpu_id}')

    def check_task_finished_in_previous_launch(self, task_index: int) -> bool:
        log_path = self.task_list[task_index].log_path

        if log_path.exists():
            if (
                not (log_path / 'task_info.json').exists() or # task not initialized
                not (log_path / 'error.txt').exists() or
                (log_path / 'uncompleted.txt').exists() or    # task not finish
                ((log_path / 'error.txt').exists() and (log_path / 'error.txt').stat().st_size > 0) # error ocurred
            ):
                task_finished_flag = False
            else:
                task_finished_flag = True
        else:
            task_finished_flag = False

        return task_finished_flag

    def launch_task_process(self, task_index: int, gpu_id: int):
        command =\
            f'HF_EVALUATE_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '+\
            f'PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES="{gpu_id}" ' +\
            f'accelerate launch --main_process_port 0 {(Path(__file__).parent / "Finetune_And_Benchmark.py").as_posix()} --task_index={task_index}'

        log_path = self.task_list[task_index].log_path
        log_path.mkdir(parents = True, exist_ok = True)
        with open(log_path / 'error.txt', 'w', encoding = 'utf8') as error_file:
            with open(log_path / 'output.txt', 'w', encoding = 'utf8') as output_file:
                subprocess.run(command, shell = True, stdout = output_file, stderr = error_file)



if __name__ == '__main__':
    Launcher().launcher_logic()

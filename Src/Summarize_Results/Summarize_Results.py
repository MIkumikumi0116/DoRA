import sys
import json
import numpy
import shutil
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append((Path(__file__).parent.parent / 'Components').as_posix())
from Global_Config import LOG_BASE_PATH, Model_Name_Enum, Dataset_Name_Enum



MIN_METRIC   = 0 # Data below this value are considered outliers and are not used for graphing and calculating correlation coefficients
LORE_RANK    = 8 # must be an integer
MODEL_NAME   = 'Roberta_Base'
DATASET_NAME = 'Squad_V2'



assert MODEL_NAME   in Model_Name_Enum.__members__,   f'invalid model name {MODEL_NAME}'
assert DATASET_NAME in Dataset_Name_Enum.__members__, f'invalid dataset name {DATASET_NAME}'
assert type(LORE_RANK) is int, 'LORE_RANK must be an integer'



class Summarizer:
    def summarize_logic(self):
        self.summarize_init()
        self.get_log_path()
        multiple_dataset_found_flag = self.load_log_data()
        self.calculate_statistics()
        self.save_statistics()

        if multiple_dataset_found_flag:
            print('found multiple datasets, will not draw correlation graph and calculate correlation score')
            return

        self.init_correlation_dataframe()
        self.draw_correlation_graph()
        self.calculate_correlation_score()

    def summarize_init(self):
        self.log_path_list: list[Path] = [] # declaration, set in get_log_path()

        self.score_dict = {} # {
                             #     checkpoint_tag: str :
                             #     {
                             #         metric_name: str : [metric_value_from_different_seed: float]
                             #     }
                             # }

        self.avg_dict  = {}  # {checkpoint_tag_and_metric_name: str : average_metric_value: float}
        self.med_dict  = {}
        self.std_dict  = {}
        self.max_dict  = {}
        self.min_dict  = {}
        self.diff_dict = {}
        self.raw_dict  = {}

        self.correlation_dataframe = None # for drawing correlation graph and calculating correlation score, only enable when there is only one model and one dataset
        self.model_name            = None
        self.dataset_name          = None
        self.metric_name_list      = []

    def get_log_path(self):
        log_path_list = []
        for log_path in (LOG_BASE_PATH / MODEL_NAME / f'Lora_Rank={LORE_RANK}' / DATASET_NAME).iterdir():
            log_path_list.append(log_path)

        self.log_path_list = log_path_list

    def load_log_data(self) -> bool:
        '''return True if multiple datasets are found, which means the correlation score will not be calculate'''

        dataset_set = set()
        for log_path in tqdm(self.log_path_list):
            checkpoint_tag = log_path.name

            if (log_path / 'uncompleted.txt').exists():
                continue

            with open(log_path / 'full_record.json', 'r', encoding = 'utf8') as file:
                finetune_result_dict = json.load(file)


            checkpoint_tag_list = checkpoint_tag.split('--')
            checkpoint_tag      = '--'.join(checkpoint_tag_list[: 1] + checkpoint_tag_list[2: ]) # remove 'seed'
            dataset_set.add((checkpoint_tag_list[0], checkpoint_tag_list[1])) # model name, dataset name

            if checkpoint_tag not in self.score_dict:
                self.score_dict[checkpoint_tag] = {}

            for metric_name, metric_list in finetune_result_dict['eval_metric_dict'].items():
                if metric_name not in self.score_dict[checkpoint_tag]:
                    self.score_dict[checkpoint_tag][metric_name] = []

                if 'prune finished' in metric_list: # for adaptive lora, only log the best metric value after pruning
                    prune_finished_index = metric_list.index('prune finished')
                    self.score_dict[checkpoint_tag][metric_name].append(max(metric_list[prune_finished_index + 1 :]))
                else:
                    self.score_dict[checkpoint_tag][metric_name].append(max(metric_list))


        if len(dataset_set) > 1:
            return True
        else:
            return False

    def calculate_statistics(self):
        def sort_by_checkpoint_tag(checkpoint_tag):
            checkpoint_tag_list = checkpoint_tag.split('--')
            for i in range(len(checkpoint_tag_list)):
                if '=' in checkpoint_tag_list[i]:
                    checkpoint_tag_list[i] = eval(checkpoint_tag_list[i].split('=')[1])

            return checkpoint_tag_list

        sorted_checkpoint_tag = sorted(self.score_dict.keys(), key = sort_by_checkpoint_tag)

        for checkpoint_tag in sorted_checkpoint_tag:
            for metric_name in self.score_dict[checkpoint_tag]:
                self.avg_dict [checkpoint_tag + '--' + metric_name] = numpy.average(self.score_dict[checkpoint_tag][metric_name])
                self.med_dict [checkpoint_tag + '--' + metric_name] = numpy.median( self.score_dict[checkpoint_tag][metric_name])
                self.std_dict [checkpoint_tag + '--' + metric_name] = numpy.std(    self.score_dict[checkpoint_tag][metric_name])
                self.max_dict [checkpoint_tag + '--' + metric_name] = max(          self.score_dict[checkpoint_tag][metric_name])
                self.min_dict [checkpoint_tag + '--' + metric_name] = min(          self.score_dict[checkpoint_tag][metric_name])
                self.diff_dict[checkpoint_tag + '--' + metric_name] = max(          self.score_dict[checkpoint_tag][metric_name]) - min(self.score_dict[checkpoint_tag][metric_name])
                self.raw_dict [checkpoint_tag + '--' + metric_name] = self.score_dict[checkpoint_tag][metric_name]

                self.avg_dict [checkpoint_tag + '--' + metric_name] = round(self.avg_dict [checkpoint_tag + '--' + metric_name], 4)
                self.med_dict [checkpoint_tag + '--' + metric_name] = round(self.med_dict [checkpoint_tag + '--' + metric_name], 4)
                self.std_dict [checkpoint_tag + '--' + metric_name] = round(self.std_dict [checkpoint_tag + '--' + metric_name], 4)
                self.max_dict [checkpoint_tag + '--' + metric_name] = round(self.max_dict [checkpoint_tag + '--' + metric_name], 4)
                self.min_dict [checkpoint_tag + '--' + metric_name] = round(self.min_dict [checkpoint_tag + '--' + metric_name], 4)
                self.diff_dict[checkpoint_tag + '--' + metric_name] = round(self.diff_dict[checkpoint_tag + '--' + metric_name], 4)

        # pad tag names to the same length
        max_tag_len = max(len(tag) for tag in self.avg_dict.keys())
        for tag in list(self.avg_dict.keys()):
            self.avg_dict [tag.rjust(max_tag_len)] = self.avg_dict. pop(tag)
            self.med_dict [tag.rjust(max_tag_len)] = self.med_dict. pop(tag)
            self.std_dict [tag.rjust(max_tag_len)] = self.std_dict. pop(tag)
            self.max_dict [tag.rjust(max_tag_len)] = self.max_dict. pop(tag)
            self.min_dict [tag.rjust(max_tag_len)] = self.min_dict. pop(tag)
            self.diff_dict[tag.rjust(max_tag_len)] = self.diff_dict.pop(tag)

    def save_statistics(self):
        summary_dict = {
            'avg' : self.avg_dict,
            'diff': self.diff_dict,
            'med' : self.med_dict,
            'std' : self.std_dict,
            'max' : self.max_dict,
            'min' : self.min_dict,
            'raw' : self.raw_dict
        }

        with open(Path(__file__).parent / 'summary.json', 'w', encoding = 'utf8') as file:
            json.dump(summary_dict, file, indent = 4, ensure_ascii = False)

    def init_correlation_dataframe(self):
        avg_data_dict = {tag: metric_value for tag, metric_value in self.avg_dict.items() if metric_value >= MIN_METRIC}
        for tag in avg_data_dict.keys():
            if self.model_name   is None:
                self.model_name   = tag.split('--')[0].strip()
            if self.dataset_name is None:
                self.dataset_name = tag.split('--')[1].strip()

            metric_name = tag.split('--')[-1].strip()
            if metric_name not in self.metric_name_list:
                self.metric_name_list.append(metric_name)

        extracted_data = []
        hyper_param_tag_to_extracted_data_index_dict = {}
        for tag, metric_value in avg_data_dict.items():
            hyper_param_list = []
            for hyper_param_str in tag.split('--'):
                if '=' in hyper_param_str:
                    hyper_param_list.append(hyper_param_str.split('='))

            metric_name     = tag.split('--')[-1].strip()
            hyper_param_tag = '--'.join(f'{name}={value}' for name, value in hyper_param_list)
            if hyper_param_tag not in hyper_param_tag_to_extracted_data_index_dict:
                hyper_param_dict = {}
                for hyper_param_name, hyper_param_value in hyper_param_list:
                    hyper_param_dict[hyper_param_name] = eval(hyper_param_value)

                hyper_param_dict[metric_name] = metric_value
                extracted_data.append(hyper_param_dict)
                hyper_param_tag_to_extracted_data_index_dict[hyper_param_tag] = len(extracted_data) - 1
            else:
                extracted_data_index = hyper_param_tag_to_extracted_data_index_dict[hyper_param_tag]
                hyper_param_dict     = extracted_data[extracted_data_index]
                hyper_param_dict[metric_name] = metric_value

        self.correlation_dataframe = pd.DataFrame(extracted_data)

    def draw_correlation_graph(self):
        for metric_name in self.metric_name_list:
            row_index      = self.correlation_dataframe[metric_name].notna()
            temp_dataframe = self.correlation_dataframe[row_index]

            exclude_metric_name_list = [metric_name_ for metric_name_ in self.metric_name_list if metric_name_ != metric_name]
            temp_dataframe = temp_dataframe.drop(columns = exclude_metric_name_list)

            sns.set(style = 'whitegrid')
            palette   = sns.color_palette('Set2', len(temp_dataframe.columns) - 1)
            fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (20, 12))
            for column_index, column_name in enumerate(temp_dataframe.columns[:-1]):
                row_index = column_index // 3
                col_index = column_index %  3

                sns.boxplot(data = temp_dataframe, x = column_name, y = metric_name, palette = palette, ax = axes[row_index, col_index])
                axes[row_index, col_index].set_ylabel(metric_name)

            for i in range(len(temp_dataframe.columns) - 1, 6):
                fig.delaxes(axes.flatten()[i])

            plt.tight_layout()
            plt.subplots_adjust(top = 0.95)
            plt.savefig(Path(__file__).parent / f'{self.model_name}_{self.dataset_name}_{metric_name}_graph.png')
            plt.close()

            print(f'max {metric_name}: {max(temp_dataframe[metric_name])}')
            print(f'\Hyper Parameter with max {metric_name}: {temp_dataframe[temp_dataframe[metric_name] == max(temp_dataframe[metric_name])]}')
            print('')

    def calculate_correlation_score(self):
        print('\n')

        for metric_name in self.metric_name_list:
            row_index      = self.correlation_dataframe[metric_name].notna()
            temp_dataframe = self.correlation_dataframe[row_index]

            pearson = temp_dataframe.corr(method = 'pearson')[metric_name]
            pearson.pop(metric_name)
            for hyper_param_name, pearson_value in pearson.items():
                print(f'{self.model_name}_{self.dataset_name}_{metric_name} {hyper_param_name}: {round(pearson_value, 4)}')

            print('\n')



if __name__ == '__main__':
    Summarizer().summarize_logic()

import sys
import time
import json
import pickle
import datasets
import multiprocessing
import numpy as np
from tqdm import tqdm
from pathlib import Path
from aenum import Enum, NoAlias
from typing import Callable, Union, Any
from transformers import AutoTokenizer
from datasets.utils.logging import disable_progress_bar

sys.path.append(Path(__file__).parent.as_posix())
from Tokenize_Config import Model_Dataset_Length_Config_Enum

sys.path.append((Path(__file__).parent.parent / 'Components').as_posix())
from Utils_And_Config import current_time
from Global_Config import (
    TOKENIZED_DATASET_BASE_PATH,
    Model_Name_Enum,
    Problem_Type_Enum,
    Model_Path_Enum,
    Dataset_Name_Enum,
    Dataset_Problem_Type_Enum,
    Dataset_Path_Enum,
    Dataset_Key_Enum
)



class Glue_Tokenizer:
    def tokenize_logic(self, model_name: Model_Name_Enum, dataset_name: Dataset_Name_Enum, task_index: int, task_num: int):
        disable_progress_bar()

        self.model_name: str   = model_name.value
        self.dataset_name: str = dataset_name.value
        self.task_index        = task_index
        self.task_num          = task_num

        assert self.dataset_name.startswith('Glue_'), f'{self.dataset_name} is not a Glue dataset'

        self.tokenizer_init()
        self.tokenizer_run()
        self.tokenizer_save_and_cleanup()

    def tokenizer_init(self):
        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} tokenization initializing')

        dataset_path        = Dataset_Path_Enum[self.dataset_name].value
        self.dataset        = datasets.load_dataset(*dataset_path)

        model_path          = Model_Path_Enum[self.model_name].value
        self.tokenizer      = AutoTokenizer.from_pretrained(model_path)

        self.sentence_key_1 = Dataset_Key_Enum[self.dataset_name].value[0]
        self.sentence_key_2 = Dataset_Key_Enum[self.dataset_name].value[1]
        self.seq_length     = Model_Dataset_Length_Config_Enum[self.model_name].value[self.dataset_name].value[0]

        self.output_path    = TOKENIZED_DATASET_BASE_PATH / self.model_name / self.dataset_name
        self.output_path.mkdir(parents = True, exist_ok = True)

        self.train_dataset_dict = { # declaration, value is set in tokenizer_run
            'input_ids'     : None,
            'attention_mask': None,
            'label'         : None
        }
        self.dev_dataset_dict   = {
            'input_ids'     : None,
            'attention_mask': None,
            'label'         : None
        }

        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} tokenization initialized')

    def tokenizer_run(self):
        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} tokenizing')

        self.process_dataset_split('train')
        self.process_dataset_split('dev')

        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} tokenized')

    def tokenizer_save_and_cleanup(self):
        self.save_dataset()
        self.save_statistical_data()

    def process_dataset_split(self, split: str): # split: 'train' or 'dev'
        assert split in ['train', 'dev'], f"split should be 'train' or 'dev', but got {split}"

        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} {split} dataset tokenizing')

        if split == 'train':
            dataset_split = self.dataset['train']
        else:
            if   self.dataset_name == 'Glue_Mnli_Matched':
                dataset_split = self.dataset['validation_matched']
            elif self.dataset_name == 'Glue_Mnli_Mismatched':
                dataset_split = self.dataset['validation_mismatched']
            else:
                dataset_split = self.dataset['validation']

        text = (dataset_split[self.sentence_key_1],) if self.sentence_key_2 is None else (dataset_split[self.sentence_key_1], dataset_split[self.sentence_key_2])
        tokenize_result = self.tokenizer(
            *text,
            padding        = 'max_length',
            max_length     = self.seq_length,
            truncation     = 'longest_first',
            return_tensors = 'np'
        )

        dataset_dict_split = self.train_dataset_dict if split == 'train' else self.dev_dataset_dict
        dataset_dict_split['input_ids']      = tokenize_result['input_ids']
        dataset_dict_split['attention_mask'] = tokenize_result['attention_mask']
        dataset_dict_split['label']          = np.array(dataset_split['label'])

        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} {split} dataset tokenized')

    def save_dataset(self):
        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} dataset saving')

        for split_name, dataset_split_dict in zip(['train', 'dev'], [self.train_dataset_dict, self.dev_dataset_dict]):
            for save_name, save_value in dataset_split_dict.items():
                memmap = np.memmap(
                    self.output_path / f'{split_name}_{save_name}.npy',
                    dtype =
                        np.float32 if
                        Dataset_Name_Enum[self.dataset_name] == Dataset_Name_Enum.Glue_Stsb and save_name == 'label'
                        else np.int64,
                    shape = save_value.shape,
                    mode  = 'w+'
                )
                memmap[:] = save_value[:]
                memmap.flush()

        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} dataset saved')

    def save_statistical_data(self):
        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} statistical data summarizing')

        train_sentence_length = np.sum(self.train_dataset_dict['attention_mask'], axis = 1)
        dev_sentence_length   = np.sum(self.dev_dataset_dict['attention_mask'],   axis = 1)

        statistical_data = {
            'train_num'        : 0,
            'dev_num'          : 0,

            'train_max_len'    : 0,
            'train_min_len'    : 0,
            'train_avg_len'    : 0,
            'train_total_token': 0,

            'dev_max_len'      : 0,
            'dev_min_len'      : 0,
            'dev_avg_len'      : 0,
            'dev_total_token'  : 0
        }

        statistical_data['train_num']         = len(train_sentence_length)
        statistical_data['dev_num']           = len(dev_sentence_length)

        statistical_data['train_max_len']     = train_sentence_length.max().item()
        statistical_data['train_min_len']     = train_sentence_length.min().item()
        statistical_data['train_avg_len']     = train_sentence_length.mean().item()
        statistical_data['train_avg_len']     = round(statistical_data['train_avg_len'], 4)
        statistical_data['train_total_token'] = train_sentence_length.sum().item()

        statistical_data['dev_max_len']       = dev_sentence_length.max().item()
        statistical_data['dev_min_len']       = dev_sentence_length.min().item()
        statistical_data['dev_avg_len']       = dev_sentence_length.mean().item()
        statistical_data['dev_avg_len']       = round(statistical_data['dev_avg_len'], 4)
        statistical_data['dev_total_token']   = dev_sentence_length.sum().item()

        with open(self.output_path / 'statistical_data.json', 'w', encoding = 'utf-8') as file:
            json.dump(statistical_data, file, indent = 4, ensure_ascii = False)

        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} statistical data summarized')

class Squad_Tokenizer:
    def tokenize_logic(self, model_name: Model_Name_Enum, dataset_name: Dataset_Name_Enum, task_index: int, task_num: int):
        disable_progress_bar()

        self.model_name: str   = model_name.value
        self.dataset_name: str = dataset_name.value
        self.task_index        = task_index
        self.task_num          = task_num

        assert self.dataset_name.startswith('Squad_'), f'{self.dataset_name} is not a Squad dataset'

        self.tokenizer_init()
        self.tokenizer_run()
        self.tokenizer_save_and_cleanup()

    def tokenizer_init(self):
        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} tokenization initializing')

        dataset_path       = Dataset_Path_Enum[self.dataset_name].value
        self.dataset       = datasets.load_dataset(*dataset_path)

        model_path         = Model_Path_Enum[self.model_name].value
        self.tokenizer     = AutoTokenizer.from_pretrained(model_path)

        self.question_key  = Dataset_Key_Enum[self.dataset_name].value[0]
        self.context_key   = Dataset_Key_Enum[self.dataset_name].value[1]
        self.answer_key    = Dataset_Key_Enum[self.dataset_name].value[2]
        self.id_key        = Dataset_Key_Enum[self.dataset_name].value[3]
        self.seq_length    = Model_Dataset_Length_Config_Enum[self.model_name].value[self.dataset_name].value[0]
        self.stride_length = Model_Dataset_Length_Config_Enum[self.model_name].value[self.dataset_name].value[1]

        self.output_path   = TOKENIZED_DATASET_BASE_PATH / self.model_name / self.dataset_name
        self.output_path.mkdir(parents = True, exist_ok = True)

        self.train_dataset_dict = { # declaration, value is set in tokenizer_run
            'input_ids'     : None,
            'attention_mask': None,
            'label'         : None
        }
        self.dev_dataset_dict   = {
            'input_ids'     : None,
            'attention_mask': None,
            'label'         : None
        }

        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} tokenization initialized')

    def tokenizer_run(self):
        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} tokenizing')

        processed_dataset = self.dataset.map(
            self.dataset_split_map_preprocess,
            batched = True,
            num_proc = 16,
            remove_columns = self.dataset.column_names['train'],
            load_from_cache_file = False,
            keep_in_memory = True
        )

        self.process_dataset_split(processed_dataset, 'train')
        self.process_dataset_split(processed_dataset, 'dev')

        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} tokenized')

    def tokenizer_save_and_cleanup(self):
        self.save_dataset()
        self.save_statistical_data()

    def dataset_split_map_preprocess(self, dataset_batch: dict[str, list]) -> dict[str, list]:
        tokenize_result = self.tokenizer(
            [text.strip() for text in dataset_batch[self.question_key]],
            dataset_batch[self.context_key],
            truncation                = 'only_second',
            padding                   = 'max_length',
            max_length                = self.seq_length,
            stride                    = self.stride_length,
            return_overflowing_tokens = True,
            return_offsets_mapping    = True
        )

        tokenize_result['label'] = self.process_dataset_split_label(tokenize_result, dataset_batch)
        tokenize_result.pop('offset_mapping')
        tokenize_result.pop('overflow_to_sample_mapping')

        return tokenize_result

    def process_dataset_split_label(self, tokenize_result: dict[str, list], dataset_batch: dict[str, list]) -> list:
        label_list = []
        for seq_index, offset_mapping in enumerate(tokenize_result['offset_mapping']): # offset_mapping is a list of tuples, each tuple is the start and end character index of the corresponding token list.
            sample_index    = tokenize_result['overflow_to_sample_mapping'][seq_index] # One example can give several spans, this is the index of the example containing this span of text.
            answer_dict     = dataset_batch[self.answer_key][sample_index]
            token_type_list = tokenize_result.sequence_ids(seq_index) # Grab the sequence corresponding to that example (to know what is the context and what is the question).

            answer_start_and_end_token_index = self.find_answer_token_index_range(offset_mapping, token_type_list, answer_dict)

            answer_start_char_index = [0]  if answer_start_and_end_token_index == (0, 0) else answer_dict['answer_start']
            answer_text             = [''] if answer_start_and_end_token_index == (0, 0) else answer_dict['text']

            label_list.append({
                'id'                              : dataset_batch[self.id_key][sample_index],
                'answer_start_and_end_token_index': answer_start_and_end_token_index,
                'answer_json'                     : json.dumps({'answer_start': answer_start_char_index, 'text': answer_text}),
                'offset_mapping'                  : offset_mapping,
                'context_text'                    : dataset_batch[self.context_key][sample_index]
            })

        return label_list

    def find_answer_token_index_range(self, offset_mapping: list[int], token_type_list: list[Union[int, None]], answer_dict: dict) -> tuple[int]:
        if len(answer_dict['answer_start']) == 0: # If no answers are given, set the (0, 0) as answer.
            answer_start_and_end_token_index = (0, 0)
        else:
            context_start_token_index = min(index for index, token_type_id in enumerate(token_type_list) if token_type_id == 1)
            context_end_token_index   = max(index for index, token_type_id in enumerate(token_type_list) if token_type_id == 1)
            answer_start_char_index   = answer_dict['answer_start'][0]
            answer_end_char_index     = answer_start_char_index + len(answer_dict['text'][0])

            if  offset_mapping[context_start_token_index][0] > answer_start_char_index or \
                offset_mapping[context_end_token_index  ][1] < answer_end_char_index:
                answer_start_and_end_token_index = (0, 0) # If the answer is out of the context (or out of sequence), set (0, 0) as answer.
            else:
                before_answer_token_index_list = [token_index for token_index in range(context_start_token_index, context_end_token_index + 1) if offset_mapping[token_index][1] <= answer_start_char_index]
                after_answer_token_index_list  = [token_index for token_index in range(context_start_token_index, context_end_token_index + 1) if offset_mapping[token_index][0] >= answer_end_char_index]
                answer_start_token_index       = before_answer_token_index_list[-1] + 1 if len(before_answer_token_index_list) > 0 else context_start_token_index
                answer_end_token_index         = after_answer_token_index_list[0]   - 1 if len(after_answer_token_index_list)  > 0 else context_end_token_index

                answer_start_and_end_token_index = (answer_start_token_index, answer_end_token_index)

        return answer_start_and_end_token_index

    def process_dataset_split(self, processed_dataset: datasets.Dataset, split :str):
        assert split in ['train', 'dev'], f"split should be 'train' or 'dev', but got {split}"
        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} {split} dataset tokenizing')

        processed_dataset_split = processed_dataset['train'] if split == 'train' else processed_dataset['validation']
        dataset_dict_split      = self.train_dataset_dict    if split == 'train' else self.dev_dataset_dict

        if self.dataset_name == Dataset_Name_Enum.Squad_V1.value:
            non_null_item_index_list = [index for index, label in enumerate(processed_dataset_split['label']) if label['answer_start_and_end_token_index'] != (0, 0)]
            processed_dataset_split  = processed_dataset_split.select(non_null_item_index_list)

        dataset_dict_split['input_ids']      = np.array(processed_dataset_split['input_ids'])
        dataset_dict_split['attention_mask'] = np.array(processed_dataset_split['attention_mask'])
        dataset_dict_split['label']          = processed_dataset_split['label']

        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} {split} dataset tokenized')

    def save_dataset(self):
        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} dataset saving')

        for split_name, dataset_dict in zip(['train', 'dev'], [self.train_dataset_dict, self.dev_dataset_dict]):
            for save_name, save_value in dataset_dict.items():
                if save_name != 'label':
                    memmap = np.memmap(
                        self.output_path / f'{split_name}_{save_name}.npy',
                        dtype = np.int64,
                        shape = save_value.shape,
                        mode  = 'w+'
                    )
                    memmap[:] = save_value[:]
                    memmap.flush()
                else:
                    with open(self.output_path / f'{split_name}_{save_name}.pkl', 'wb') as file:
                        pickle.dump(save_value, file)

        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} dataset saved')

    def save_statistical_data(self):
        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} statistical data summarizing')

        train_sentence_length = np.sum(self.train_dataset_dict['attention_mask'], axis = 1)
        dev_sentence_length   = np.sum(self.dev_dataset_dict['attention_mask'],   axis = 1)

        statistical_data = {
            'train_num'        : 0,
            'dev_num'          : 0,

            'train_max_len'    : 0,
            'train_min_len'    : 0,
            'train_avg_len'    : 0,
            'train_total_token': 0,

            'dev_max_len'      : 0,
            'dev_min_len'      : 0,
            'dev_avg_len'      : 0,
            'dev_total_token'  : 0
        }

        statistical_data['train_num']         = len(train_sentence_length)
        statistical_data['dev_num']           = len(dev_sentence_length)

        statistical_data['train_max_len']     = train_sentence_length.max().item()
        statistical_data['train_min_len']     = train_sentence_length.min().item()
        statistical_data['train_avg_len']     = train_sentence_length.mean().item()
        statistical_data['train_avg_len']     = round(statistical_data['train_avg_len'], 4)
        statistical_data['train_total_token'] = train_sentence_length.sum().item()

        statistical_data['dev_max_len']       = dev_sentence_length.max().item()
        statistical_data['dev_min_len']       = dev_sentence_length.min().item()
        statistical_data['dev_avg_len']       = dev_sentence_length.mean().item()
        statistical_data['dev_avg_len']       = round(statistical_data['dev_avg_len'], 4)
        statistical_data['dev_total_token']   = dev_sentence_length.sum().item()

        with open(self.output_path / 'statistical_data.json', 'w', encoding = 'utf-8') as file:
            json.dump(statistical_data, file, indent = 4, ensure_ascii = False)

        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} statistical data summarized')

class Summarization_Tokenizer:
    def tokenize_logic(self, model_name: Model_Name_Enum, dataset_name: Dataset_Name_Enum, task_index: int, task_num: int):
        disable_progress_bar()

        self.model_name: str   = model_name.value
        self.dataset_name: str = dataset_name.value
        self.task_index        = task_index
        self.task_num          = task_num

        summary_dataset_name_list = [dataset_name for dataset_name in Dataset_Name_Enum.__members__ if Dataset_Problem_Type_Enum[dataset_name].value == Problem_Type_Enum.Summarization]
        assert self.dataset_name in summary_dataset_name_list, f'{self.dataset_name} is not a summarization dataset'

        self.tokenizer_init()
        self.tokenizer_run()
        self.tokenizer_save_and_cleanup()

    def tokenizer_init(self):
        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} tokenization initializing')

        dataset_path        = Dataset_Path_Enum[self.dataset_name].value
        self.dataset        = datasets.load_dataset(*dataset_path)

        model_path          = Model_Path_Enum[self.model_name].value
        self.tokenizer      = AutoTokenizer.from_pretrained(model_path)

        self.text_key       = Dataset_Key_Enum[self.dataset_name].value[0]
        self.summary_key    = Dataset_Key_Enum[self.dataset_name].value[1]
        self.text_length    = Model_Dataset_Length_Config_Enum[self.model_name].value[self.dataset_name].value[0]
        self.summary_length = Model_Dataset_Length_Config_Enum[self.model_name].value[self.dataset_name].value[1]

        self.output_path    = TOKENIZED_DATASET_BASE_PATH / self.model_name / self.dataset_name
        self.output_path.mkdir(parents = True, exist_ok = True)

        self.train_dataset_dict = { # declaration, value is set in tokenizer_run
            'input_ids'     : None,
            'attention_mask': None,
            'label'         : None
        }
        self.dev_dataset_dict   = {
            'input_ids'     : None,
            'attention_mask': None,
            'label'         : None
        }

        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} tokenization initialized')

    def tokenizer_run(self):
        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} tokenizing')

        self.process_dataset_split('train')
        self.process_dataset_split('dev')

        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} tokenized')

    def tokenizer_save_and_cleanup(self):
        self.save_dataset()
        self.save_statistical_data()

    def process_dataset_split(self, split :str):
        assert split in ['train', 'dev'], f"split should be 'train' or 'dev', but got {split}"
        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} {split} dataset tokenizing')

        dataset_split   = self.dataset['train'] if split == 'train' else self.dataset['validation']

        text_list       = [text.strip() for text in dataset_split[self.text_key]]
        tokenize_result = self.tokenizer(
            text_list,
            padding        = 'max_length',
            max_length     = self.text_length,
            truncation     = 'longest_first',
            return_tensors = 'np'
        )

        dataset_dict_split = self.train_dataset_dict if split == 'train' else self.dev_dataset_dict
        dataset_dict_split['input_ids']      = tokenize_result['input_ids']
        dataset_dict_split['attention_mask'] = tokenize_result['attention_mask']

        summary_list    = [summary.strip() for summary in dataset_split[self.summary_key]]
        tokenize_result = self.tokenizer(
            text_target    = summary_list,
            padding        = 'max_length',
            max_length     = self.summary_length,
            truncation     = 'longest_first',
            return_tensors = 'np'
        )

        label_array = tokenize_result['input_ids']
        label_array[label_array == self.tokenizer.pad_token_id] = -100
        dataset_dict_split['label'] = label_array

        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} {split} dataset tokenized')

    def save_dataset(self):
        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} dataset saving')

        for split_name, dataset_dict in zip(['train', 'dev'], [self.train_dataset_dict, self.dev_dataset_dict]):
            for save_name, save_value in dataset_dict.items():
                memmap = np.memmap(
                    self.output_path / f'{split_name}_{save_name}.npy',
                    dtype = np.int64,
                    shape = save_value.shape,
                    mode  = 'w+'
                )
                memmap[:] = save_value[:]
                memmap.flush()

        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} dataset saved')

    def save_statistical_data(self):
        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} statistical data summarizing')

        train_sentence_length = np.sum(self.train_dataset_dict['attention_mask'], axis = 1)
        dev_sentence_length   = np.sum(self.dev_dataset_dict['attention_mask'],   axis = 1)

        statistical_data = {
            'train_num'        : 0,
            'dev_num'          : 0,

            'train_max_len'    : 0,
            'train_min_len'    : 0,
            'train_avg_len'    : 0,
            'train_total_token': 0,

            'dev_max_len'      : 0,
            'dev_min_len'      : 0,
            'dev_avg_len'      : 0,
            'dev_total_token'  : 0
        }

        statistical_data['train_num']         = len(train_sentence_length)
        statistical_data['dev_num']           = len(dev_sentence_length)

        statistical_data['train_max_len']     = train_sentence_length.max().item()
        statistical_data['train_min_len']     = train_sentence_length.min().item()
        statistical_data['train_avg_len']     = train_sentence_length.mean().item()
        statistical_data['train_avg_len']     = round(statistical_data['train_avg_len'], 4)
        statistical_data['train_total_token'] = train_sentence_length.sum().item()

        statistical_data['dev_max_len']       = dev_sentence_length.max().item()
        statistical_data['dev_min_len']       = dev_sentence_length.min().item()
        statistical_data['dev_avg_len']       = dev_sentence_length.mean().item()
        statistical_data['dev_avg_len']       = round(statistical_data['dev_avg_len'], 4)
        statistical_data['dev_total_token']   = dev_sentence_length.sum().item()

        with open(self.output_path / 'statistical_data.json', 'w', encoding = 'utf-8') as file:
            json.dump(statistical_data, file, indent = 4, ensure_ascii = False)

        print(f'{current_time()} {self.task_index + 1}/{self.task_num} {self.dataset_name} statistical data summarized')



def main():
    class Enum_Function_Wrapper:
        def __init__(self, func: Callable):
            self.func = func

        def __call__(self, *args: Any, **kwargs: Any):
            return self.func(*args, **kwargs)

    class Dataset_Tokenize_Func_Enum(Enum):
        _settings_ = NoAlias

        Glue_Rte: Callable             = Enum_Function_Wrapper(Glue_Tokenizer().tokenize_logic)
        Glue_Mrpc: Callable            = Enum_Function_Wrapper(Glue_Tokenizer().tokenize_logic)
        Glue_Stsb: Callable            = Enum_Function_Wrapper(Glue_Tokenizer().tokenize_logic)
        Glue_Cola: Callable            = Enum_Function_Wrapper(Glue_Tokenizer().tokenize_logic)
        Glue_Sst2: Callable            = Enum_Function_Wrapper(Glue_Tokenizer().tokenize_logic)
        Glue_Qnli: Callable            = Enum_Function_Wrapper(Glue_Tokenizer().tokenize_logic)
        Glue_Mnli_Matched: Callable    = Enum_Function_Wrapper(Glue_Tokenizer().tokenize_logic)
        Glue_Mnli_Mismatched: Callable = Enum_Function_Wrapper(Glue_Tokenizer().tokenize_logic)
        Glue_Qqp: Callable             = Enum_Function_Wrapper(Glue_Tokenizer().tokenize_logic)
        Squad_V1: Callable             = Enum_Function_Wrapper(Squad_Tokenizer().tokenize_logic)
        Squad_V2: Callable             = Enum_Function_Wrapper(Squad_Tokenizer().tokenize_logic)
        Xsum: Callable                 = Enum_Function_Wrapper(Summarization_Tokenizer().tokenize_logic)
        Cnn_Daily_Mail: Callable       = Enum_Function_Wrapper(Summarization_Tokenizer().tokenize_logic)

    assert set(Dataset_Name_Enum.__members__) >= set(Dataset_Tokenize_Func_Enum.__members__), 'Dataset_Tokenize_Func_Enum should be a subset of Dataset_Name_Enum'

    tokenize_arg_list = []
    for model_dataset_length_enum in Model_Dataset_Length_Config_Enum:
        model_name = model_dataset_length_enum.name

        for dataset_length_enum in model_dataset_length_enum.value:
            dataset_name = dataset_length_enum.name

            tokenize_arg_list.append((model_name, dataset_name))

    process_list = []
    task_num = len(tokenize_arg_list)
    for task_index, (model_name, dataset_name) in enumerate(tokenize_arg_list):
        tokenize_func = Dataset_Tokenize_Func_Enum[dataset_name].value

        process = multiprocessing.Process(
            target = tokenize_func,
            args   = (Model_Name_Enum[model_name], Dataset_Name_Enum[dataset_name], task_index, task_num)
        )
        process.start()
        process_list.append(process)

    progress_bar = tqdm(total = len(process_list))
    while len(process_list) > 0:
        process_index = 0
        while process_index < len(process_list):
            if not process_list[process_index].is_alive():
                process_list.pop(process_index)
                progress_bar.update(1)
            else:
                process_index += 1

        time.sleep(1)

    print(f'{current_time()} All dataset tokenized')



if __name__ == '__main__':
    main()

import sys
import nltk
import json
import torch
import scipy
import shutil
import logging
import argparse
import evaluate
import warnings
import transformers
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import Callable
from accelerate import Accelerator

sys.path.append(Path(__file__).parent.as_posix())
from Build_Finetune_Config_List import build_finetune_config_list
from Finetune_Utils import Full_Finetune_Model_Wrapper, Fixed_Lora_Model_Wrapper, Adaptive_Lora_Model_Wrapper, Dataset_Type_Enum

sys.path.append((Path(__file__).parent.parent / 'Components').as_posix())
from Utils_And_Config import current_time
from Global_Config import (
    Dataset_Name_Enum,
    Finetune_Type_Enum,
    Problem_Type_Enum,
    Model_Path_Enum,
    Dataset_Problem_Type_Enum,
    Problem_Type_Model_Type_Enum,
)



class Trainer:
    def trainer_logic(self):
        finetune_config_list = build_finetune_config_list()

        self.task_index  = self.parse_task_index()
        self.task_num    = len(finetune_config_list)
        self.config      = finetune_config_list[self.task_index]
        self.step_num    = 0
        self.record_dict = {
            'train_loss'               : [],
            'train_true_label_loss'    : [],
            'train_regularization_loss': [],
            'eval_loss'                : [],
            'eval_true_label_loss'     : [],
            'eval_regularization_loss' : [],

            'train_metric_dict': {report_metric_name: [] for report_metric_name in self.config.report_metric_list},
            'eval_metric_dict' : {report_metric_name: [] for report_metric_name in self.config.report_metric_list},
        }

        self.accelerator = Accelerator(split_batches = True, gradient_accumulation_steps = self.config.grad_accumulation_step)

        self.init_task()

        self.eval_method = evaluate.load(self.config.eval_metric)
        self.train_dataloader, self.dev_dataloader = self.init_data()
        self.max_step = len(self.train_dataloader) * self.config.epoch
        self.problem_type = Dataset_Problem_Type_Enum[self.config.dataset_name.value].value

        self.model, self.optimizer, self.lr_scheduler, self.tokenizer = self.init_model()
        self.model, self.optimizer, self.lr_scheduler, self.train_dataloader, self.dev_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, self.train_dataloader, self.dev_dataloader
        )

        self.checkpoint_step_num, self.checkpoint_epoch_num = self.load_checkpoint() # how many epochs have trained in checkpoint, and how many steps in the last epoch have trained, -1 for no checkpoint found

        self.train_forward_func = self.init_train_forward_func()
        self.eval_forward_func  = self.init_eval_forward_func()

        for epoch in range(self.config.epoch):
            if epoch < self.checkpoint_epoch_num:
                continue

            self.train(epoch)
            self.eval(epoch)
            self.save_record(epoch)

        self.cleanup_task()

    def parse_task_index(self) -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument('--task_index', type = int, required = True)
        args = parser.parse_args()
        task_index = args.task_index

        return task_index

    def init_task(self):
        transformers.set_seed(self.config.seed)
        torch.backends.cudnn.benchmark     = False
        torch.backends.cudnn.deterministic = True

        transformers.logging.set_verbosity_error()
        warnings.filterwarnings('ignore', category = scipy.stats.ConstantInputWarning)
        logging.basicConfig(level = logging.ERROR)
        evaluate_logger = logging.getLogger('evaluate.loading')
        evaluate_logger.setLevel(logging.ERROR)

        if self.accelerator.is_local_main_process:
            self.config.log_path.mkdir(       parents = True, exist_ok = True)
            self.config.checkpoint_path.mkdir(parents = True, exist_ok = True)

            uncompleted_indicate_file_path = self.config.log_path / 'uncompleted.txt'
            with open(uncompleted_indicate_file_path, 'w', encoding = 'utf8') as file:
                file.write('This file is used for indicating the training is not completed, will be deleted automatically when the training is completed.')
            if not (self.config.log_path / 'train_log.txt').exists():
                with open(self.config.log_path / 'train_log.txt', 'w', encoding = 'utf8') as _:
                    pass # create empty file
            if not (self.config.log_path / 'eval_log.txt').exists():
                with open(self.config.log_path / 'eval_log.txt', 'w', encoding = 'utf8') as _:
                    pass

        self.accelerator.wait_for_everyone()
        self.accelerator.print(f'{current_time()} task {self.task_index + 1}/{self.task_num} {self.config.task_tag} task initialized')

    def init_data(self) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        assert self.config.batch_size % self.accelerator.num_processes == 0, f'batch size {self.config.batch_size} must be divided by num gpus {self.accelerator.num_processes}'

        dataset_type     = Dataset_Type_Enum[self.config.dataset_name.value].value
        train_dataset    = dataset_type(self.config.model_name, self.config.dataset_name, 'train')
        dev_dataset      = dataset_type(self.config.model_name, self.config.dataset_name,   'dev')

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle      = True,
            batch_size   = self.config.batch_size
        )
        dev_dataloader   = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size   = self.config.batch_size
        )

        self.accelerator.wait_for_everyone()
        self.accelerator.print(f'{current_time()} task {self.task_index + 1}/{self.task_num} {self.config.task_tag} data initialized')

        return train_dataloader, dev_dataloader

    def init_model(self) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, transformers.PreTrainedTokenizer]:
        model_type   = Problem_Type_Model_Type_Enum[self.problem_type.value].value
        model_path   = Model_Path_Enum[self.config.model_name.value].value
        model_config = transformers.AutoConfig.from_pretrained(model_path, num_labels = self.config.num_label) if self.config.num_label != 0 else transformers.AutoConfig.from_pretrained(model_path)
        model        = model_type.from_pretrained(model_path, config = model_config)

        if   self.config.finetune_type == Finetune_Type_Enum.Fixed_Lora:
            model = Fixed_Lora_Model_Wrapper(self.config, model)
        elif self.config.finetune_type == Finetune_Type_Enum.Adaptive_Lora:
            model = Adaptive_Lora_Model_Wrapper(self.config, model, max_step = self.max_step, main_process_flag = self.accelerator.is_local_main_process)
        elif self.config.finetune_type == Finetune_Type_Enum.Full_Finetune:
            model = Full_Finetune_Model_Wrapper(self.config, model)
        else:
            raise NotImplementedError(f'unknown finetune type {self.config.finetune_type}')

        optimizer    = self.config.optimizer_type(model.trainable_param(), lr = self.config.lr, weight_decay = self.config.weight_decay)
        lr_scheduler = transformers.get_scheduler(
            name               = self.config.lr_scheduler_type,
            optimizer          = optimizer,
            num_warmup_steps   = int(self.max_step * self.config.warmup_ratio),
            num_training_steps = self.max_step,
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

        if self.accelerator.is_local_main_process:
            config_dict = self.config.to_dict()
            config_dict.pop('checkpoint_path')
            config_dict.pop('log_path')
            with open(self.config.log_path / 'task_info.json', 'w', encoding = 'utf8') as file:
                json.dump(config_dict, file, indent = 4, ensure_ascii = False)

        self.accelerator.wait_for_everyone()
        self.accelerator.print(f'{current_time()} task {self.task_index + 1}/{self.task_num} {self.config.task_tag} model initialized')

        return model, optimizer, lr_scheduler, tokenizer

    def save_checkpoint(self):
        checkpoint_path      = self.config.checkpoint_path / 'checkpoint'
        temp_checkpoint_path = self.config.checkpoint_path / 'temp_checkpoint'

        self.accelerator.save_state(output_dir = temp_checkpoint_path)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_local_main_process:
            if self.config.finetune_type == Finetune_Type_Enum.Adaptive_Lora:
                self.accelerator.unwrap_model(self.model).save_state(temp_checkpoint_path)

            with open(temp_checkpoint_path / 'step_num.json',    'w', encoding = 'utf8') as file:
                json.dump(self.step_num,    file, indent = 4, ensure_ascii = False)

            torch.save(self.record_dict, temp_checkpoint_path / 'record_dict.pt')

            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            temp_checkpoint_path.rename(checkpoint_path)

        self.accelerator.wait_for_everyone()
        self.accelerator.print(f'{current_time()} task {self.task_index + 1}/{self.task_num} {self.config.task_tag} checkpoint saved')

    def load_checkpoint(self) -> int:
        checkpoint_path = self.config.checkpoint_path / 'checkpoint'

        if checkpoint_path.exists():
            self.accelerator.load_state(checkpoint_path)

            if self.config.finetune_type == Finetune_Type_Enum.Adaptive_Lora:
                self.accelerator.unwrap_model(self.model).load_state(checkpoint_path)

            with open(checkpoint_path / 'step_num.json', 'r', encoding = 'utf8') as file:
                self.step_num        = json.load(file)
                checkpoint_step_num  = self.step_num %  len(self.train_dataloader) # step have trained in last epoch in checkpoint, or -1 if no checkpoint found, should be skipped
                checkpoint_epoch_num = self.step_num // len(self.train_dataloader) # epoch have trained in checkpoint, or -1 if no checkpoint found, should be skipped

            self.record_dict = torch.load(checkpoint_path / 'record_dict.pt')

            self.accelerator.wait_for_everyone()
            self.accelerator.print(f'{current_time()} task {self.task_index + 1}/{self.task_num} {self.config.task_tag} checkpoint loaded, trained {checkpoint_epoch_num} epoch, {checkpoint_step_num} step')
        else:
            checkpoint_step_num  = -1
            checkpoint_epoch_num = -1

            self.accelerator.wait_for_everyone()
            self.accelerator.print(f'{current_time()} task {self.task_index + 1}/{self.task_num} {self.config.task_tag} no checkpoint found')

        return checkpoint_step_num, checkpoint_epoch_num

    def init_train_forward_func(self) -> Callable:
        '''prediction and reference will be gathered in forward function, while loss will not be gathered in forward function for backward compatibility'''

        def seq_classification_train_forward_func(input_ids: torch.Tensor, attention_mask: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            input_dict = {
                'input_ids':      input_ids,
                'attention_mask': attention_mask,
                'labels':         label
            }
            logits, loss = self.model(input_dict)

            pred = torch.argmax(logits, dim = 1)
            pred, label = self.accelerator.gather_for_metrics((pred, label))

            return loss, pred, label

        def seq_regression_train_forward_func(input_ids: torch.Tensor, attention_mask: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            input_dict = {
                'input_ids':      input_ids,
                'attention_mask': attention_mask,
                'labels':         label
            }
            logits, loss = self.model(input_dict)

            pred = logits.squeeze()
            pred, label = self.accelerator.gather_for_metrics((pred, label))

            return loss, pred, label

        def question_answering_train_forward_func(input_ids: torch.Tensor, attention_mask: torch.Tensor, label: dict) -> tuple[torch.Tensor, list[dict], list[dict]]:
            input_dict = {
                'input_ids':       input_ids,
                'attention_mask':  attention_mask,
                'start_positions': label['answer_start_and_end_token_index'][0],
                'end_positions':   label['answer_start_and_end_token_index'][1]
            }
            (start_logits, end_logits), loss = self.model(input_dict)


            start_logits = self.accelerator.pad_across_processes(start_logits, dim = 1, pad_index = -10000)
            end_logits   = self.accelerator.pad_across_processes(end_logits,   dim = 1, pad_index = -10000)

            start_logits, end_logits, batch_index = self.accelerator.gather_for_metrics((start_logits, end_logits, label['index']))

            start_logits = start_logits.detach().cpu().numpy()
            end_logits   = end_logits  .detach().cpu().numpy()

            batch_pred_dict = {}
            for seq_index, sample_index in enumerate(batch_index):
                seq_start_logits = start_logits[seq_index]
                seq_end_logits   = end_logits  [seq_index]

                label_dict     = self.train_dataloader.dataset[sample_index][2]
                offset_mapping = label_dict['offset_mapping']

                special_token_index_list = [offset_index for offset_index, offset_pair in enumerate(offset_mapping) if offset_pair == [0, 0]] # [CLS] question [SEP] context [SEP] [PAD]
                bound_pair_list          = [] # between bounds are question/context tokens, context tokens are the second one
                for special_token_index in range(1, len(special_token_index_list)):
                    if special_token_index_list[special_token_index] != special_token_index_list[special_token_index - 1] + 1:
                        bound_pair_list.append((special_token_index_list[special_token_index - 1], special_token_index_list[special_token_index]))

                context_start_token_index = bound_pair_list[1][0] + 1
                context_end_token_index   = bound_pair_list[1][1] - 1

                seq_pred_list          = []
                start_token_index_list = np.argsort(seq_start_logits)[-1 : -self.config.question_answering_max_answer_num - 1 : -1].tolist()
                end_token_index_list   = np.argsort(seq_end_logits)  [-1 : -self.config.question_answering_max_answer_num - 1 : -1].tolist()
                for start_token_index in start_token_index_list:
                    for end_token_index in end_token_index_list:
                        if (
                            start_token_index < context_start_token_index or
                            end_token_index   > context_end_token_index or
                            end_token_index   < start_token_index or
                            end_token_index   - start_token_index + 1 > self.config.question_answering_max_answer_length
                        ):
                            continue

                        context     = label_dict['context_text']
                        char_offset = (offset_mapping[start_token_index][0], offset_mapping[end_token_index][1])
                        pred_text   = context[char_offset[0]: char_offset[1]]

                        seq_pred_list.append({
                            'pred_text': pred_text,
                            'score'    : seq_start_logits[start_token_index] + seq_end_logits[end_token_index]
                        })


                seq_pred_list = sorted(seq_pred_list, key = lambda pred_dict: pred_dict['score'], reverse = True)
                seq_pred_list = seq_pred_list[: self.config.question_answering_max_answer_num]
                if len(seq_pred_list) == 0:
                    seq_pred_list.append({'pred_text': '', 'score': 0.0}) # In the very rare edge case we have not a single non-null prediction

                if self.config.dataset_name != Dataset_Name_Enum.Squad_V2:
                    batch_pred_dict[label_dict['id']] = seq_pred_list[0]['pred_text']
                else:
                    best_non_null_pred = seq_pred_list[0]
                    score_diff = seq_start_logits[0] + seq_end_logits[0] - best_non_null_pred['score']
                    if score_diff > self.config.question_answering_no_answer_threshold:
                        batch_pred_dict[label_dict['id']] = ''
                    else:
                        batch_pred_dict[label_dict['id']] = best_non_null_pred['pred_text']


            prediction = []
            reference  = []
            for (id, pred_text), sample_index in zip(batch_pred_dict.items(), batch_index):
                label_dict = self.train_dataloader.dataset[sample_index][2]
                prediction.append({
                    'prediction_text': pred_text,
                    'id': id
                })
                reference.append({
                    'answers': json.loads(label_dict['answer_json']),
                    'id': id
                })

                if self.config.dataset_name == Dataset_Name_Enum.Squad_V2:
                    for prediction_dict in prediction:
                        prediction_dict['no_answer_probability'] = 0

            return loss, prediction, reference

        def summarization_train_forward_func(input_ids: torch.Tensor, attention_mask: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, None, None]:
            input_dict = {
                'input_ids':      input_ids,
                'attention_mask': attention_mask,
                'labels':         label
            }
            _, loss = self.model(input_dict)

            return loss, None, None

        if   self.problem_type == Problem_Type_Enum.Seq_Classification:
            return seq_classification_train_forward_func
        elif self.problem_type == Problem_Type_Enum.Seq_Regression:
            return seq_regression_train_forward_func
        elif self.problem_type == Problem_Type_Enum.Question_Answering:
            return question_answering_train_forward_func
        elif self.problem_type == Problem_Type_Enum.Summarization:
            return summarization_train_forward_func
        else:
            raise NotImplementedError(f'unknown problem type {self.problem_type}')

    def init_eval_forward_func(self) -> Callable:
        '''prediction and reference will be gathered in forward function, while loss will not be gathered in forward function for backward compatibility'''

        def seq_classification_eval_forward_func(input_ids: torch.Tensor, attention_mask: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            input_dict = {
                'input_ids':      input_ids,
                'attention_mask': attention_mask,
                'labels':         label
            }
            logits, loss = self.model(input_dict)

            pred = torch.argmax(logits, dim = 1)
            pred, label = self.accelerator.gather_for_metrics((pred, label))

            return loss, pred, label

        def seq_regression_eval_forward_func(input_ids: torch.Tensor, attention_mask: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            input_dict = {
                'input_ids':      input_ids,
                'attention_mask': attention_mask,
                'labels':         label
            }
            logits, loss = self.model(input_dict)

            pred = logits.squeeze()
            pred, label = self.accelerator.gather_for_metrics((pred, label))

            return loss, pred, label

        def question_answering_eval_forward_func(input_ids: torch.Tensor, attention_mask: torch.Tensor, label: dict) -> tuple[torch.Tensor, list[dict], list[dict]]:
            input_dict = {
                'input_ids':       input_ids,
                'attention_mask':  attention_mask,
                'start_positions': label['answer_start_and_end_token_index'][0],
                'end_positions':   label['answer_start_and_end_token_index'][1]
            }
            (start_logits, end_logits), loss = self.model(input_dict)


            start_logits = self.accelerator.pad_across_processes(start_logits, dim = 1, pad_index = -10000)
            end_logits   = self.accelerator.pad_across_processes(end_logits,   dim = 1, pad_index = -10000)

            start_logits, end_logits, batch_index = self.accelerator.gather_for_metrics((start_logits, end_logits, label['index']))

            start_logits = start_logits.detach().cpu().numpy()
            end_logits   = end_logits  .detach().cpu().numpy()

            batch_pred_dict = {}
            for seq_index, sample_index in enumerate(batch_index):
                seq_start_logits = start_logits[seq_index]
                seq_end_logits   = end_logits  [seq_index]

                label_dict     = self.dev_dataloader.dataset[sample_index][2]
                offset_mapping = label_dict['offset_mapping']

                special_token_index_list = [offset_index for offset_index, offset_pair in enumerate(offset_mapping) if offset_pair == [0, 0]] # [CLS] question [SEP] context [SEP] [PAD]
                bound_pair_list          = [] # between bounds are question/context tokens, context tokens are the second one
                for special_token_index in range(1, len(special_token_index_list)):
                    if special_token_index_list[special_token_index] != special_token_index_list[special_token_index - 1] + 1:
                        bound_pair_list.append((special_token_index_list[special_token_index - 1], special_token_index_list[special_token_index]))

                context_start_token_index = bound_pair_list[1][0] + 1
                context_end_token_index   = bound_pair_list[1][1] - 1

                seq_pred_list          = []
                start_token_index_list = np.argsort(seq_start_logits)[-1 : -self.config.question_answering_max_answer_num - 1 : -1].tolist()
                end_token_index_list   = np.argsort(seq_end_logits)  [-1 : -self.config.question_answering_max_answer_num - 1 : -1].tolist()
                for start_token_index in start_token_index_list:
                    for end_token_index in end_token_index_list:
                        if (
                            start_token_index < context_start_token_index or
                            end_token_index   > context_end_token_index or
                            end_token_index   < start_token_index or
                            end_token_index   - start_token_index + 1 > self.config.question_answering_max_answer_length
                        ):
                            continue

                        context     = label_dict['context_text']
                        char_offset = (offset_mapping[start_token_index][0], offset_mapping[end_token_index][1])
                        pred_text   = context[char_offset[0]: char_offset[1]]

                        seq_pred_list.append({
                            'pred_text': pred_text,
                            'score'    : seq_start_logits[start_token_index] + seq_end_logits[end_token_index]
                        })


                seq_pred_list = sorted(seq_pred_list, key = lambda pred_dict: pred_dict['score'], reverse = True)
                seq_pred_list = seq_pred_list[: self.config.question_answering_max_answer_num]
                if len(seq_pred_list) == 0:
                    seq_pred_list.append({'pred_text': '', 'score': 0.0}) # In the very rare edge case we have not a single non-null prediction

                if self.config.dataset_name != Dataset_Name_Enum.Squad_V2:
                    batch_pred_dict[label_dict['id']] = seq_pred_list[0]['pred_text']
                else:
                    best_non_null_pred = seq_pred_list[0]
                    score_diff = seq_start_logits[0] + seq_end_logits[0] - best_non_null_pred['score']
                    if score_diff > self.config.question_answering_no_answer_threshold:
                        batch_pred_dict[label_dict['id']] = ''
                    else:
                        batch_pred_dict[label_dict['id']] = best_non_null_pred['pred_text']


            prediction = []
            reference  = []
            for (id, pred_text), sample_index in zip(batch_pred_dict.items(), batch_index):
                label_dict = self.dev_dataloader.dataset[sample_index][2]
                prediction.append({
                    'prediction_text': pred_text,
                    'id': id
                })
                reference.append({
                    'answers': json.loads(label_dict['answer_json']),
                    'id': id
                })

                if self.config.dataset_name == Dataset_Name_Enum.Squad_V2:
                    for prediction_dict in prediction:
                        prediction_dict['no_answer_probability'] = 0

            return loss, prediction, reference

        def summarization_eval_forward_func(input_ids: torch.Tensor, attention_mask: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, list[str], list[str]]:
            input_dict = {
                'input_ids':      input_ids,
                'attention_mask': attention_mask,
                'labels':         label
            }
            _, loss = self.model(input_dict)

            model = self.accelerator.unwrap_model(self.model).model
            pred  = model.generate(
                inputs         = input_ids,
                attention_mask = attention_mask,
                max_new_tokens = self.config.generate_max_length,
                num_beams      = self.config.generate_beam_size
            )

            pred = self.accelerator.pad_across_processes(pred, dim = 1, pad_index = self.tokenizer.pad_token_id)
            label[label == -100] = self.tokenizer.pad_token_id
            pred, label = self.accelerator.gather_for_metrics((pred, label))

            pred  = self.tokenizer.batch_decode(pred,  skip_special_tokens = True)
            label = self.tokenizer.batch_decode(label, skip_special_tokens = True)

            pred  = ['\n'.join(nltk.sent_tokenize(pred. strip())) for pred  in pred ]
            label = ['\n'.join(nltk.sent_tokenize(label.strip())) for label in label]

            return loss, pred, label

        if   self.problem_type == Problem_Type_Enum.Seq_Classification:
            return seq_classification_eval_forward_func
        elif self.problem_type == Problem_Type_Enum.Seq_Regression:
            return seq_regression_eval_forward_func
        elif self.problem_type == Problem_Type_Enum.Question_Answering:
            return question_answering_eval_forward_func
        elif self.problem_type == Problem_Type_Enum.Summarization:
            return summarization_eval_forward_func
        else:
            raise NotImplementedError(f'unknown problem type {self.problem_type}')

    def train(self, epoch: int):
        progress_bar      = tqdm(
            total         = len(self.train_dataloader),
            desc          = f'{current_time()} task {self.task_index + 1}/{self.task_num} {self.config.task_tag[:30] + "..."} train epoch {epoch + 1}/{self.config.epoch}',
            disable       = not self.accelerator.is_local_main_process,
            file          = sys.stdout,
            dynamic_ncols = True
        )

        if self.checkpoint_step_num != -1:
            dataloader = self.accelerator.skip_first_batches(self.train_dataloader, self.checkpoint_step_num)
            for _ in range(self.checkpoint_step_num):
                progress_bar.update()
            self.checkpoint_step_num = -1
        else:
            dataloader = self.train_dataloader

        self.model.train()
        for input_ids, attention_mask, label in dataloader:
            with self.accelerator.accumulate(self.model):
                self.step_num += 1
                progress_bar.set_description_str(f'{current_time()} task {self.task_index + 1}/{self.task_num} {self.config.task_tag[:30] + "..."} train epoch {epoch + 1}/{self.config.epoch}')
                progress_bar.update()

                max_seq_len    = attention_mask.sum(dim = 1).max().item()
                input_ids      = input_ids     [:, :max_seq_len]
                attention_mask = attention_mask[:, :max_seq_len]

                true_label_loss, prediction, reference = self.train_forward_func(input_ids, attention_mask, label)

                regularization_loss = self.accelerator.unwrap_model(self.model).regularization_loss(self.step_num)
                loss = true_label_loss + self.config.regularization_loss_alpha * regularization_loss
                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(self.accelerator.unwrap_model(self.model).trainable_param(), self.config.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

                loss, true_label_loss, regularization_loss = self.gather_loss(loss, true_label_loss, regularization_loss)
                self.record_dict['train_loss'               ].append(loss)
                self.record_dict['train_true_label_loss'    ].append(true_label_loss)
                self.record_dict['train_regularization_loss'].append(regularization_loss)

                if self.problem_type != Problem_Type_Enum.Summarization:
                    self.eval_method.add_batch(predictions = prediction, references = reference)

                if self.config.finetune_type == Finetune_Type_Enum.Adaptive_Lora:
                    prune_finish_flag = self.accelerator.unwrap_model(self.model).prune_lora_scaler(self.step_num)
                    if prune_finish_flag:
                        self.log_prune_finish()

                if self.step_num % self.config.save_interval == 0:
                    self.save_checkpoint()


        if self.problem_type != Problem_Type_Enum.Summarization:
            eval_result = self.eval_method.compute()
            for report_metric_name in self.config.report_metric_list:
                self.record_dict['train_metric_dict'][report_metric_name].append(eval_result[report_metric_name])

    def log_prune_finish(self):
        info_str = f'{current_time()} task {self.task_index + 1}/{self.task_num} {self.config.task_tag} prune finished'

        self.accelerator.print(info_str)
        if self.accelerator.is_local_main_process:
            with open(self.config.log_path / 'train_log.txt', 'a', encoding = 'utf8') as file:
                file.write(info_str + '\n')
            with open(self.config.log_path /  'eval_log.txt', 'a', encoding = 'utf8') as file:
                file.write(info_str + '\n')

        self.record_dict['train_loss'               ].append('prune finished')
        self.record_dict['train_true_label_loss'    ].append('prune finished')
        self.record_dict['train_regularization_loss'].append('prune finished')
        for report_metric_name in self.config.report_metric_list:
            self.record_dict['train_metric_dict'][report_metric_name].append('prune finished')

        self.record_dict['eval_loss'               ].append('prune finished')
        self.record_dict['eval_true_label_loss'    ].append('prune finished')
        self.record_dict['eval_regularization_loss'].append('prune finished')
        for report_metric_name in self.config.report_metric_list:
            self.record_dict['eval_metric_dict'][report_metric_name].append('prune finished')

    def eval(self, epoch: int):
        loss_record                = 0
        true_label_loss_record     = 0
        regularization_loss_record = 0

        progress_bar      = tqdm(
            total         = len(self.dev_dataloader),
            desc          = f'{current_time()} task {self.task_index + 1}/{self.task_num} {self.config.task_tag[:30] + "..."} eval  epoch {epoch + 1}/{self.config.epoch}',
            disable       = not self.accelerator.is_local_main_process,
            file          = sys.stdout,
            dynamic_ncols = True
        )

        self.model.eval()
        with torch.inference_mode():
            regularization_loss_ = self.accelerator.unwrap_model(self.model).regularization_loss(self.step_num) # regularization loss will not change during evaluation, so we can compute it only once

            for input_ids, attention_mask, label in self.dev_dataloader:
                progress_bar.set_description_str(f'{current_time()} task {self.task_index + 1}/{self.task_num} {self.config.task_tag[:30] + "..."} eval  epoch {epoch + 1}/{self.config.epoch}')
                progress_bar.update()

                max_seq_len    = attention_mask.sum(dim = 1).max().item()
                input_ids      = input_ids     [:, :max_seq_len]
                attention_mask = attention_mask[:, :max_seq_len]

                true_label_loss, prediction, reference = self.eval_forward_func(input_ids, attention_mask, label)

                loss = true_label_loss + self.config.regularization_loss_alpha * regularization_loss_
                loss, true_label_loss, regularization_loss = self.gather_loss(loss, true_label_loss, regularization_loss_)
                loss_record                += loss
                true_label_loss_record     += true_label_loss
                regularization_loss_record += regularization_loss

                self.eval_method.add_batch(predictions = prediction, references = reference)


        self.record_dict['eval_loss'               ].append(loss_record                / len(self.dev_dataloader))
        self.record_dict['eval_true_label_loss'    ].append(true_label_loss_record     / len(self.dev_dataloader))
        self.record_dict['eval_regularization_loss'].append(regularization_loss_record / len(self.dev_dataloader))
        eval_result = self.eval_method.compute()
        for report_metric_name in self.config.report_metric_list:
            self.record_dict['eval_metric_dict'][report_metric_name].append(eval_result[report_metric_name])

    def gather_loss(self, loss: torch.FloatTensor, true_label_loss: torch.FloatTensor, regularization_loss: torch.FloatTensor) -> tuple[float, float, float]:
        loss                = loss.detach().clone() # prevent memory leak
        true_label_loss     = true_label_loss.detach().clone()
        regularization_loss = regularization_loss.detach().clone()

        loss, true_label_loss, regularization_loss = self.accelerator.gather((loss, true_label_loss, regularization_loss))

        loss                = loss.mean()
        true_label_loss     = true_label_loss.mean()
        regularization_loss = regularization_loss.mean()

        return loss, true_label_loss, regularization_loss

    def save_record(self, epoch: int):
        for record_list in [
            self.record_dict['train_loss'],
            self.record_dict['train_true_label_loss'],
            self.record_dict['train_regularization_loss'],
            self.record_dict['eval_loss'],
            self.record_dict['eval_true_label_loss'],
            self.record_dict['eval_regularization_loss']
        ]:
            for index in range(len(record_list) - 1, -1, -1):
                if type(record_list[index]) == torch.Tensor:
                    record_list[index] = record_list[index].item()
                elif type(record_list[index]) == str:
                    continue
                else:
                    break

        train_loss_list = [loss for loss in self.record_dict['train_loss'][-len(self.train_dataloader):] if isinstance(loss, float)]
        train_loss_mean = sum(train_loss_list) / len(self.train_dataloader)
        train_info_str  = f'{current_time()} task {self.task_index + 1}/{self.task_num} {self.config.task_tag} train epoch {epoch + 1}/{self.config.epoch} loss: {train_loss_mean:.4f}'
        if self.problem_type != Problem_Type_Enum.Summarization:
            for report_metric_name in self.config.report_metric_list:
                train_info_str += f' {report_metric_name}: {self.record_dict["train_metric_dict"][report_metric_name][-1] :.4f}'

        self.accelerator.print(train_info_str)
        if self.accelerator.is_local_main_process:
            with open(self.config.log_path / 'train_log.txt', 'a', encoding = 'utf8') as file:
                file.write(train_info_str + '\n')


        eval_loss_list = [loss for loss in self.record_dict['eval_loss'][-len(self.dev_dataloader):] if isinstance(loss, float)]
        eval_loss_mean = sum(eval_loss_list) / len(self.dev_dataloader)
        eval_info_str  = f'{current_time()} task {self.task_index + 1}/{self.task_num} {self.config.task_tag} eval  epoch {epoch + 1}/{self.config.epoch} loss: {eval_loss_mean:.4f}'
        for report_metric_name in self.config.report_metric_list:
            eval_info_str += f' {report_metric_name}: {self.record_dict["eval_metric_dict"][report_metric_name][-1] :.4f}'

        self.accelerator.print(eval_info_str)
        if self.accelerator.is_local_main_process:
            with open(self.config.log_path / 'eval_log.txt', 'a', encoding = 'utf8') as file:
                file.write(eval_info_str + '\n')


        with open(self.config.log_path / 'full_record.json', 'w', encoding = 'utf8') as file:
            json.dump(self.record_dict, file, indent = 4, ensure_ascii = False)


        if self.accelerator.is_local_main_process:
            column_num = 3 + len(self.config.report_metric_list)
            _, axs = plt.subplots(2, column_num, figsize = (10 * column_num, 8))

            plot_data_list = (
                [('train_loss',                   [loss  for loss  in self.record_dict['train_loss']                            if not isinstance(loss,  str)])] + # remove 'prune finished'
                [('train_true_label_loss',        [loss  for loss  in self.record_dict['train_true_label_loss']                 if not isinstance(loss,  str)])] +
                [('train_regularization_loss',    [loss  for loss  in self.record_dict['train_regularization_loss']             if not isinstance(loss,  str)])] +
                ([(f'train_{report_metric_name}', [score for score in self.record_dict['train_metric_dict'][report_metric_name] if not isinstance(score, str)]) for report_metric_name in self.config.report_metric_list] if self.problem_type != Problem_Type_Enum.Summarization else []) +
                [('eval_loss',                    [loss  for loss  in self.record_dict['eval_loss'               ]              if not isinstance(loss,  str)])] +
                [('eval_true_label_loss',         [loss  for loss  in self.record_dict['eval_true_label_loss'    ]              if not isinstance(loss,  str)])] +
                [('eval_regularization_loss',     [loss  for loss  in self.record_dict['eval_regularization_loss']              if not isinstance(loss,  str)])] +
                [(f'eval_{report_metric_name}',   [score for score in self.record_dict['eval_metric_dict'][report_metric_name]  if not isinstance(score, str)]) for report_metric_name in self.config.report_metric_list]
            )
            for ax, (record_name, record_list) in zip(axs.flat, plot_data_list):
                x_label = [index for index in range(len(record_list))]
                ax.plot(x_label, record_list)
                ax.set_title(record_name)

            plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
            plt.savefig(self.config.log_path / 'log.png')
            plt.clf()
            plt.close()

        self.accelerator.wait_for_everyone()
        self.accelerator.print(f'{current_time()} task {self.task_index + 1}/{self.task_num} {self.config.task_tag} epoch {epoch + 1}/{self.config.epoch} record saved')

    def cleanup_task(self):
        info_str = f'{current_time()} task {self.task_index + 1}/{self.task_num} {self.config.task_tag} finished'
        self.accelerator.print(info_str)
        if self.accelerator.is_local_main_process:
            uncompleted_indicate_file_path = self.config.log_path / 'uncompleted.txt'
            uncompleted_indicate_file_path.unlink()
            with open(self.config.log_path / 'train_log.txt', 'a', encoding = 'utf8') as file:
                file.write(info_str + '\n')
            with open(self.config.log_path /  'eval_log.txt', 'a', encoding = 'utf8') as file:
                file.write(info_str + '\n')


        if self.config.finetune_type == Finetune_Type_Enum.Adaptive_Lora:
            self.accelerator.wait_for_everyone()
            model = self.accelerator.unwrap_model(self.model)
            pruned_param_num = model.pruned_param_num()

            info_str = f'{current_time()} task {self.task_index + 1}/{self.task_num} {self.config.task_tag} pruned_param_num: {pruned_param_num}'
            self.accelerator.print(info_str)
            if self.accelerator.is_local_main_process:
                with open(self.config.log_path / 'train_log.txt', 'a', encoding = 'utf8') as file:
                    file.write(info_str + '\n')
                with open(self.config.log_path /  'eval_log.txt', 'a', encoding = 'utf8') as file:
                    file.write(info_str + '\n')


        self.accelerator.wait_for_everyone()
        self.accelerator.free_memory()



if __name__ == '__main__':
    Trainer().trainer_logic()

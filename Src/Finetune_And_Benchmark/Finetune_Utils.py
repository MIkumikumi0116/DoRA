import re
import gc
import sys
import json
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from aenum import Enum, NoAlias

sys.path.append(Path(__file__).parent.as_posix())
from Finetune_Config import Finetune_Config

sys.path.append((Path(__file__).parent.parent / 'Components').as_posix())
from Global_Config import (
    TOKENIZED_DATASET_BASE_PATH,
    Model_Name_Enum,
    Model_Architecture_Enum,
    Model_Type_Enum,
    Problem_Type_Enum,
    Dataset_Name_Enum,
    Dataset_Problem_Type_Enum
)

sys.path.append((Path(__file__).parent.parent / 'Dataset_Tokenize').as_posix())
from Tokenize_Config import Model_Dataset_Length_Config_Enum



class Glue_Dataset(torch.utils.data.Dataset):
    def __init__(self, model_name: Model_Name_Enum, dataset_name: Dataset_Name_Enum, split: str): # split: 'train' or 'dev'
        super().__init__()

        self.model_name: str   = model_name.value
        self.dataset_name: str = dataset_name.value
        self.split             = split

        assert self.dataset_name.startswith('Glue_'), f'{self.dataset_name} is not a Glue dataset'

        self.dataset_path = TOKENIZED_DATASET_BASE_PATH / self.model_name / self.dataset_name
        with open(self.dataset_path / 'statistical_data.json', 'r', encoding = 'utf8') as file:
            statistical_data = json.load(file)

        self.dataset_num = statistical_data[f'{split}_num']
        self.seq_length  = Model_Dataset_Length_Config_Enum[self.model_name].value[self.dataset_name].value[0]

        self.loaded_step           = None # declaration, value will be assigned in reload_dataset()
        self.input_ids_memmap      = None
        self.attention_mask_memmap = None
        self.label_memmap          = None

        self.reload_dataset()

    def reload_dataset(self):
        # reload dataset to free cache memory of memmap
        self.loaded_step           = 0
        self.input_ids_memmap      = np.memmap(self.dataset_path / f'{self.split}_input_ids.npy',      mode = 'r', dtype = np.int64, shape = (self.dataset_num, self.seq_length))
        self.attention_mask_memmap = np.memmap(self.dataset_path / f'{self.split}_attention_mask.npy', mode = 'r', dtype = np.int64, shape = (self.dataset_num, self.seq_length))
        self.label_memmap          = np.memmap(
            self.dataset_path / f'{self.split}_label.npy',
            mode  = 'r',
            dtype = np.int64 if Dataset_Name_Enum[self.dataset_name] != Dataset_Name_Enum.Glue_Stsb else np.float32,
            shape = (self.dataset_num,)
        )

        gc.collect()

    def __len__(self):
        return self.dataset_num

    def __getitem__(self, index):
        self.loaded_step += 1

        if self.loaded_step == 2000:
            self.reload_dataset()

        return (
            torch.tensor(self.input_ids_memmap[index]),
            torch.tensor(self.attention_mask_memmap[index]),
            torch.tensor(self.label_memmap[index])
        )

class Question_Answering_Dataset(torch.utils.data.Dataset):
    def __init__(self, model_name: Model_Name_Enum, dataset_name: Dataset_Name_Enum, split: str): # split: 'train' or 'dev'
        super().__init__()

        self.model_name: str   = model_name.value
        self.dataset_name: str = dataset_name.value
        self.split             = split

        assert self.dataset_name.startswith('Squad_'), f'{self.dataset_name} is not a Squad dataset'

        self.dataset_path = TOKENIZED_DATASET_BASE_PATH / self.model_name / self.dataset_name
        with open(self.dataset_path / 'statistical_data.json', 'r', encoding = 'utf8') as file:
            statistical_data = json.load(file)

        self.dataset_num = statistical_data[f'{split}_num']
        self.seq_length  = Model_Dataset_Length_Config_Enum[self.model_name].value[self.dataset_name].value[0]

        self.loaded_step           = None # declaration, value will be assigned in reload_dataset()
        self.input_ids_memmap      = None
        self.attention_mask_memmap = None

        with open(self.dataset_path / f'{split}_label.pkl', 'rb') as file:
            self.label_list = pickle.load(file)

        self.reload_dataset()

    def reload_dataset(self):
        # reload dataset to free cache memory of memmap
        self.loaded_step           = 0
        self.input_ids_memmap      = np.memmap(self.dataset_path / f'{self.split}_input_ids.npy',      mode = 'r', dtype = np.int64, shape = (self.dataset_num, self.seq_length))
        self.attention_mask_memmap = np.memmap(self.dataset_path / f'{self.split}_attention_mask.npy', mode = 'r', dtype = np.int64, shape = (self.dataset_num, self.seq_length))

        gc.collect()

    def __len__(self):
        return self.dataset_num

    def __getitem__(self, index):
        self.loaded_step += 1

        if self.loaded_step == 2000:
            self.reload_dataset()

        label = self.label_list[index]
        label['index'] = index

        return (
            torch.tensor(self.input_ids_memmap[index]),
            torch.tensor(self.attention_mask_memmap[index]),
            label
        )

class Summarization_Dataset(torch.utils.data.Dataset):
    def __init__(self, model_name: Model_Name_Enum, dataset_name: Dataset_Name_Enum, split: str):
        super().__init__()

        self.model_name: str   = model_name.value
        self.dataset_name: str = dataset_name.value
        self.split             = split

        summary_dataset_name_list = [dataset_name for dataset_name in Dataset_Name_Enum.__members__ if Dataset_Problem_Type_Enum[dataset_name].value == Problem_Type_Enum.Summarization]
        assert self.dataset_name in summary_dataset_name_list, f'{self.dataset_name} is not a summarization dataset'

        self.dataset_path = TOKENIZED_DATASET_BASE_PATH / self.model_name / self.dataset_name
        with open(self.dataset_path / 'statistical_data.json', 'r', encoding = 'utf8') as file:
            statistical_data = json.load(file)

        self.dataset_num    = statistical_data[f'{split}_num']
        self.text_length    = Model_Dataset_Length_Config_Enum[self.model_name].value[self.dataset_name].value[0]
        self.summary_length = Model_Dataset_Length_Config_Enum[self.model_name].value[self.dataset_name].value[1]

        self.loaded_step           = None # declaration, value will be assigned in reload_dataset()
        self.input_ids_memmap      = None
        self.attention_mask_memmap = None
        self.label_memmap          = None

        self.reload_dataset()

    def reload_dataset(self):
        # reload dataset to free cache memory of memmap
        self.loaded_step           = 0
        self.input_ids_memmap      = np.memmap(self.dataset_path / f'{self.split}_input_ids.npy',      mode = 'r', dtype = np.int64, shape = (self.dataset_num, self.text_length))
        self.attention_mask_memmap = np.memmap(self.dataset_path / f'{self.split}_attention_mask.npy', mode = 'r', dtype = np.int64, shape = (self.dataset_num, self.text_length))
        self.label_memmap          = np.memmap(self.dataset_path / f'{self.split}_label.npy',          mode = 'r', dtype = np.int64, shape = (self.dataset_num, self.summary_length))

        gc.collect()

    def __len__(self):
        return self.dataset_num

    def __getitem__(self, index):
        self.loaded_step += 1

        if self.loaded_step == 2000:
            self.reload_dataset()

        return (
            torch.tensor(self.input_ids_memmap[index]),
            torch.tensor(self.attention_mask_memmap[index]),
            torch.tensor(self.label_memmap[index])
        )

class Dataset_Type_Enum(Enum):
    _settings_ = NoAlias

    Glue_Cola: torch.utils.data.Dataset            = Glue_Dataset
    Glue_Mnli_Matched: torch.utils.data.Dataset    = Glue_Dataset
    Glue_Mnli_Mismatched: torch.utils.data.Dataset = Glue_Dataset
    Glue_Mrpc: torch.utils.data.Dataset            = Glue_Dataset
    Glue_Qnli: torch.utils.data.Dataset            = Glue_Dataset
    Glue_Qqp: torch.utils.data.Dataset             = Glue_Dataset
    Glue_Rte: torch.utils.data.Dataset             = Glue_Dataset
    Glue_Sst2: torch.utils.data.Dataset            = Glue_Dataset
    Glue_Stsb: torch.utils.data.Dataset            = Glue_Dataset
    Squad_V1: torch.utils.data.Dataset             = Question_Answering_Dataset
    Squad_V2: torch.utils.data.Dataset             = Question_Answering_Dataset
    Xsum: torch.utils.data.Dataset                 = Summarization_Dataset
    Cnn_Daily_Mail: torch.utils.data.Dataset       = Summarization_Dataset

assert set(Dataset_Name_Enum.__members__) >= set(Dataset_Type_Enum.__members__), 'Dataset_Type_Enum should be a subset of Dataset_Name_Enum'



class Model_Wrapper_Base(torch.nn.Module):
    def __init__(self, config: Finetune_Config, model: torch.nn.Module):
        super().__init__()
        self.config = config

        self.model = model

    def forward(self, input_dict: dict[str, torch.Tensor]) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        output = self.model(**input_dict)

        if 'logits' in output:
            logits = output.logits
        elif 'start_logits' in output and 'end_logits' in output:
            logits = (output.start_logits, output.end_logits)
        else:
            raise NotImplementedError

        loss = output.loss

        return logits, loss

    def regularization_loss(self, current_step: int) -> torch.FloatTensor:
        '''regularization_loss is only activated in adaptive lora, should be overwrite in adaptive lora wrapper, so this function just returns 0'''

        loss = torch.tensor(0, dtype = torch.float32, device = next(self.parameters()).device)

        return loss

    def trainable_param(self) -> iter:
        return (parameter for parameter in self.parameters() if parameter.requires_grad == True)

    def named_trainable_param(self) -> iter:
        return ((name, parameter) for name, parameter in self.named_parameters() if parameter.requires_grad == True)

    def trainable_param_num(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_param())



class Full_Finetune_Model_Wrapper(Model_Wrapper_Base):
    def __init__(self, config: Finetune_Config, model: torch.nn.Module):
        super().__init__(config, model)

        # full finetune model has nothing special to do



class Fixed_Lora_Linear(torch.nn.Module):
    def __init__(self, config: Finetune_Config, linear: torch.nn.Linear):
        super().__init__()
        self.config = config

        assert config.fixed_lora_rank >= 0, f'config.fixed_lora_rank must be non-negative, but got {config.fixed_lora_rank}'

        self.linear = linear
        self.linear.weight.requires_grad = False
        if linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.lora_dropout  = torch.nn.Dropout(config.lora_dropout)
        self.lora_a_linear = torch.nn.Linear(linear.in_features,     config.fixed_lora_rank,  bias = False)
        self.lora_b_linear = torch.nn.Linear(config.fixed_lora_rank, linear.out_features,     bias = False)
        self.lora_scaling  = config.lora_alpha / config.fixed_lora_rank

        torch.nn.init.kaiming_uniform_(self.lora_a_linear.weight)
        torch.nn.init.zeros_(self.lora_b_linear.weight)

    def forward(self, feature: torch.FloatTensor) -> torch.FloatTensor:
        hidden      = self.linear(feature)

        lora_hidden = self.lora_dropout(feature)
        lora_hidden = self.lora_a_linear(lora_hidden)
        lora_hidden = self.lora_b_linear(lora_hidden)
        lora_hidden = lora_hidden * self.lora_scaling

        output      = hidden + lora_hidden

        return output

class Fixed_Lora_Model_Wrapper(Model_Wrapper_Base):
    def __init__(self, config: Finetune_Config, model: torch.nn.Module):
        super().__init__(config, model)

        self.apply_fixed_lora()

    def apply_fixed_lora(self):
        for module_name, module in self.model.named_modules():
            if module_name.startswith('deberta') or module_name.startswith('roberta') or module_name.startswith('model'):
                if type(module) == torch.nn.Linear and any(linear_name in module_name for linear_name in self.config.linear_name_to_apply_lora_list):
                    name_parts  = module_name.split('.')
                    sub_module  = self.model
                    for sub_name in name_parts[: -1]:
                        sub_module = getattr(sub_module, sub_name)

                    lora_linear = Fixed_Lora_Linear(self.config, module)
                    setattr(sub_module, name_parts[-1], lora_linear)

        for param_name, param in self.model.named_parameters():
            if param_name.startswith('deberta') or param_name.startswith('roberta') or param_name.startswith('model'):
                if 'lora' in param_name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

    def trainable_param_num(self) -> int:
        '''overwrite from Model_Wrapper_Base'''

        trainable_param_num = 0
        for module in self.model.modules():
            if type(module) == Fixed_Lora_Linear:
                trainable_param_num += module.lora_a_linear.weight.numel() + module.lora_b_linear.weight.numel()

        return trainable_param_num



class Adaptive_Lora_Linear(torch.nn.Module):
    def __init__(self, config: Finetune_Config, linear: torch.nn.Linear):
        super().__init__()
        self.config = config

        self.linear = linear
        self.linear.weight.requires_grad = False
        if linear.bias is not None:
            self.linear.bias.requires_grad = False

        lora_rank = config.adaptive_lora_start_rank

        self.lora_dropout  = torch.nn.Dropout(config.lora_dropout)
        self.lora_a_linear = torch.nn.Linear(linear.in_features, lora_rank,  bias = False)
        self.lora_scaler   = torch.nn.Parameter(torch.zeros(lora_rank, dtype = torch.float32))
        self.lora_b_linear = torch.nn.Linear(lora_rank, linear.out_features, bias = False)
        self.lora_scaling  = config.lora_alpha / lora_rank

        torch.nn.init.kaiming_uniform_(self.lora_a_linear.weight)
        torch.nn.init.kaiming_uniform_(self.lora_b_linear.weight)

    def forward(self, feature: torch.FloatTensor) -> torch.FloatTensor:
        hidden      = self.linear(feature)

        lora_hidden = self.lora_dropout(feature)
        lora_hidden = self.lora_a_linear(lora_hidden)
        lora_hidden = lora_hidden * self.lora_scaler
        lora_hidden = self.lora_b_linear(lora_hidden)
        lora_hidden = lora_hidden * self.lora_scaling

        output      = hidden + lora_hidden

        return output

class Adaptive_Lora_Model_Wrapper(Model_Wrapper_Base):
    def __init__(self, config: Finetune_Config, model: torch.nn.Module, max_step: int, main_process_flag: bool):
        super().__init__(config, model)
        self.max_step                   = max_step
        self.main_process_flag          = main_process_flag
        self.sensitivity_score_dict     = {} # {name of linear: FloatTensor}
        self.finally_mask_dict          = {}

        self.apply_adaptive_lora()

    def regularization_loss(self, current_step: int) -> torch.FloatTensor:
        '''overwrite from Model_Wrapper_Base'''

        current_prune_step, max_prune_step = self.get_prune_step(current_step)
        if current_prune_step > max_prune_step: # disable the regularization loss after pruning
            return torch.tensor(0, dtype = torch.float32, device = next(self.parameters()).device)

        adaptive_lore_linear_num = 0
        sum_var = torch.tensor(0, dtype = torch.float32, device = next(self.parameters()).device)
        for module in self.model.modules():
            if type(module) == Adaptive_Lora_Linear:
                adaptive_lore_linear_num += 1

                a_linear_var = module.lora_a_linear.weight.var(dim = 1)
                b_linear_var = module.lora_b_linear.weight.var(dim = 0)
                sum_var     += a_linear_var.sum() + b_linear_var.sum()

        mean_var = sum_var / (adaptive_lore_linear_num * 2 * self.config.adaptive_lora_start_rank)

        return mean_var

    def trainable_param_num(self) -> int:
        '''overwrite from Model_Wrapper_Base'''

        trainable_param_num = 0
        for module in self.model.modules():
            if type(module) == Adaptive_Lora_Linear:
                trainable_param_num += module.lora_a_linear.weight.numel() + module.lora_scaler.numel() + module.lora_b_linear.weight.numel()

        return trainable_param_num

    def apply_adaptive_lora(self):
        for module_name, module in self.model.named_modules():
            if module_name.startswith('deberta') or module_name.startswith('roberta') or module_name.startswith('model'):
                if type(module) == torch.nn.Linear and any(target_linear_name in module_name for target_linear_name in self.config.linear_name_to_apply_lora_list):
                    name_parts  = module_name.split('.')
                    sub_module  = self.model
                    for sub_name in name_parts[: -1]:
                        sub_module = getattr(sub_module, sub_name)

                    lora_linear = Adaptive_Lora_Linear(self.config, module)
                    setattr(sub_module, name_parts[-1], lora_linear)

        for param_name, param in self.model.named_parameters():
            if param_name.startswith('deberta') or param_name.startswith('roberta') or param_name.startswith('model'):
                if 'lora' in param_name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

    def get_prune_step(self, current_step: int) -> tuple[int]:
        current_prune_step = int(current_step - self.max_step * self.config.adaptive_lora_start_prune_step_ratio)
        max_prune_step     = int(self.max_step * (1 - self.config.adaptive_lora_start_prune_step_ratio - self.config.adaptive_lora_end_prune_step_ratio))

        return current_prune_step, max_prune_step

    def prune_lora_scaler(self, current_step: int) -> bool:
        '''return True if the pruning is finished in this step, otherwise return False, even if the pruning is finished in previous step'''

        current_prune_step, max_prune_step = self.get_prune_step(current_step)

        if current_prune_step > max_prune_step:
            with torch.no_grad():
                for name, module in self.model.named_modules():
                    if type(module) == Adaptive_Lora_Linear:
                        module.lora_scaler[self.finally_mask_dict[name]] = 0

            if not (self.config.log_path / 'Adaptive_Lora_Info_Graph').exists():
                self.draw_adaptive_lora_info_graph()

            return False


        self.update_importance_score()
        if current_prune_step < 0:
            return False
        if not (current_prune_step % self.config.adaptive_lora_prune_interval_step == 0 or current_prune_step == max_prune_step):
            return False


        with torch.no_grad():
            score_dict      = {name: self.sensitivity_score_dict[name] for name in self.sensitivity_score_dict.keys()}

            all_score       = torch.cat(list(score_dict.values()))

            start_rank      = self.config.adaptive_lora_start_rank
            end_rank        = self.config.adaptive_lora_end_avg_rank
            prune_rank_rate = ((start_rank - end_rank) / start_rank) * ((current_prune_step / max_prune_step) ** 3)
            prune_rank_num  = int(all_score.numel() * prune_rank_rate)

            threshold       = torch.kthvalue(all_score, max(prune_rank_num, 1)).values
            for name, module in self.model.named_modules():
                if type(module) == Adaptive_Lora_Linear:
                    module.lora_scaler[score_dict[name] <= threshold] = 0

                    if current_prune_step == max_prune_step:
                        self.finally_mask_dict[name] = score_dict[name] <= threshold


        if current_prune_step == max_prune_step:
            return True
        else:
            return False

    def update_importance_score(self):
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if type(module) == Adaptive_Lora_Linear:
                    delta_w   = module.lora_b_linear.weight @ (module.lora_a_linear.weight * module.lora_scaler.unsqueeze(1))
                    norm_w    = delta_w.norm(p = 'fro') + self.config.adaptive_lora_eps

                    product_i = torch.einsum('ik,kj->kij', module.lora_b_linear.weight, module.lora_a_linear.weight * module.lora_scaler.unsqueeze(1))
                    norm_i    = product_i.norm(p = 'fro', dim = (1, 2))

                    score_i   = norm_i / norm_w


                    if name not in self.sensitivity_score_dict:
                        self.sensitivity_score_dict[name] = torch.zeros_like(score_i)
                        self.finally_mask_dict[name]      = torch.zeros_like(score_i)

                    self.sensitivity_score_dict[name] =      self.config.adaptive_lora_sensitivity_beta  * self.sensitivity_score_dict[name] +\
                                                        (1 - self.config.adaptive_lora_sensitivity_beta) * score_i

    def draw_adaptive_lora_info_graph(self):
        if not self.main_process_flag:
            return

        model_architecture = Model_Type_Enum[self.config.model_name.value].value
        if   model_architecture == Model_Architecture_Enum.Encoder_Only:
            hidden_layer_num = self.model.config.num_hidden_layers
        elif model_architecture == Model_Architecture_Enum.Encoder_Decoder:
            hidden_layer_num = self.model.config.encoder_layers + self.model.config.decoder_layers
        elif model_architecture == Model_Architecture_Enum.Decoder_Only:
            raise NotImplementedError
        else:
            raise ValueError(f'unknown model architecture: {model_architecture}')

        (self.config.log_path / 'Adaptive_Lora_Info_Graph').mkdir(parents = True, exist_ok = True)


        lora_rank_map = [[0 for _ in range(hidden_layer_num)] for _ in self.config.linear_name_to_apply_lora_list]
        for name, module in self.model.named_modules():
            if type(module) == Adaptive_Lora_Linear:
                layer_index = int(re.findall(r'\.(\d+)\.', name)[0])
                if model_architecture == Model_Architecture_Enum.Encoder_Decoder and 'decoder' in name:
                    layer_index += self.model.config.encoder_layers

                for linear_name_to_apply_lora_index, linear_name_to_apply_lora in enumerate(self.config.linear_name_to_apply_lora_list):
                    if linear_name_to_apply_lora in name:
                        break

                lora_rank_map[linear_name_to_apply_lora_index][layer_index] = (module.lora_scaler != 0.0).sum().item()


        if model_architecture == Model_Architecture_Enum.Encoder_Decoder:
            x_label = \
                [f'encoder_{encoder_layer_index}' for encoder_layer_index in range(self.model.config.encoder_layers)] + \
                [f'decoder_{decoder_layer_index}' for decoder_layer_index in range(self.model.config.decoder_layers)]
        else:
            x_label = [str(layer_index) for layer_index in range(self.model.config.num_hidden_layers)]

        y_label   = self.config.linear_name_to_apply_lora_list
        mean_rank = sum(sum(lora_rank_map, [])) / (len(lora_rank_map) * len(lora_rank_map[0]))

        plt.figure(figsize = (10, 6))
        plt.title(f'mean rank: {mean_rank: .4f}')
        heatmap = plt.imshow(lora_rank_map, cmap = 'viridis', aspect = 'auto')
        plt.colorbar(heatmap)
        plt.xticks(range(len(x_label)), labels = x_label)
        plt.yticks(range(len(y_label)), labels = y_label)
        for y_index in range(len(y_label)):
            for x_index in range(len(x_label)):
                plt.text(x_index, y_index, f'{lora_rank_map[y_index][x_index]}', ha = 'center', va = 'center', color = 'white')

        plt.savefig(self.config.log_path / 'Adaptive_Lora_Info_Graph' / f'lora_rank_map.png')
        plt.clf()
        plt.close()


        with open(self.config.log_path / 'Adaptive_Lora_Info_Graph' / 'lora_rank_map.json', 'w', encoding = 'utf8') as file:
            json.dump(lora_rank_map, file)

    def pruned_param_num(self) -> int:
        '''
        return the number of pruned parameters
        this function should be called after prune_lora_scaler() and before next optimizer.step()
        or the pruned lora_scaler might be recovered to prevent getting right pruned_param_num
        '''

        pruned_param_num = 0
        for module in self.model.modules():
            if type(module) == Adaptive_Lora_Linear:
                pruned_rank_num    = (module.lora_scaler != 0.0).sum().item()
                linear_a_param_num = pruned_rank_num * module.lora_a_linear.weight.shape[1]
                linear_b_param_num = pruned_rank_num * module.lora_b_linear.weight.shape[0]

                pruned_param_num  += linear_a_param_num + linear_b_param_num

        return pruned_param_num

    def save_state(self, save_folder: Path):
        prune_state_dict = {
            'sensitivity_score_dict': {name: tensor.cpu() for name, tensor in self.sensitivity_score_dict.items()},
            'finally_mask_dict'     : {name: tensor.cpu() for name, tensor in self.finally_mask_dict.items()},
        }

        torch.save(prune_state_dict, save_folder / 'prune_state_dict.pt')

    def load_state(self, save_folder: Path):
        prune_state_dict = torch.load(save_folder / 'prune_state_dict.pt')

        device = next(self.model.parameters()).device

        self.sensitivity_score_dict = {name: tensor.to(device) for name, tensor in prune_state_dict['sensitivity_score_dict'].items()}
        self.finally_mask_dict      = {name: tensor.to(device) for name, tensor in prune_state_dict['finally_mask_dict'].items()}

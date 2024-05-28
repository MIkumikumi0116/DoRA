from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaTokenizerFast,

    BartConfig,
    BartModel,
    BartTokenizer
)

config    = RobertaConfig.from_pretrained('roberta-base')
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
model     = RobertaModel.from_pretrained('roberta-base')

config    = BartConfig.from_pretrained('facebook/bart-base')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model     = BartModel.from_pretrained('facebook/bart-base')



from datasets import load_dataset, load_metric

dataset = load_dataset('glue', 'cola')
dataset = load_dataset('glue', 'mnli')
dataset = load_dataset('glue', 'mrpc')
dataset = load_dataset('glue', 'qnli')
dataset = load_dataset('glue', 'qqp')
dataset = load_dataset('glue', 'rte')
dataset = load_dataset('glue', 'sst2')
dataset = load_dataset('glue', 'stsb')

dataset = load_dataset('squad')
dataset = load_dataset('squad_v2')
dataset = load_dataset('EdinburghNLP/xsum')
dataset = load_dataset('cnn_dailymail', '3.0.0')

metric = load_metric('accuracy')
metric = load_metric('matthews_correlation')
metric = load_metric('pearsonr')
metric = load_metric('rouge')
metric = load_metric('squad')
metric = load_metric('squad_v2')



import evaluate

metric = evaluate.load('accuracy')
metric = evaluate.load('matthews_correlation')
metric = evaluate.load('pearsonr')
metric = evaluate.load('rouge')
metric = evaluate.load('squad')
metric = evaluate.load('squad_v2')

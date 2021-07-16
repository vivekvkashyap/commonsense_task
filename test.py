import jax
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, unfreeze

from typing import Any, Optional, Tuple

from transformers import (
    GPT2Config)

import transformers
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2",pad_token='<|endoftext|>') 
from datasets import load_dataset,load_metric

from model_file import FlaxGPT2ForMultipleChoice

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dataset=load_dataset('cosmos_qa')

len_test_dataset=6963

test_dataset=dataset['test'].select(range(len_test_dataset))

def preprocess(example):
    example['context&question']=example['context']+example['question']
    example['first_sentence']=[example['context&question'],example['context&question'],example['context&question'],example['context&question']]
    example['second_sentence']=example['answer0'],example['answer1'],example['answer2'],example['answer3']
    return example

test_dataset=test_dataset.map(preprocess)

def tokenize(examples):
    a=tokenizer(examples['first_sentence'],examples['second_sentence'],padding='max_length',truncation=True,max_length=256,return_tensors='jax')
    a['labels']=examples['label']
    return a

test_dataset=test_dataset.map(tokenize)

remov_col=['id', 'context', 'question', 'answer0', 'answer1', 'answer2', 'answer3', 'labels', 'context&question', 'first_sentence', 'second_sentence']

test_dataset=test_dataset.remove_columns(remov_col)

model = FlaxGPT2ForMultipleChoice.from_pretrained("flax-community/gpt2-Cosmos",input_shape=(1,4,1))

input_id=jnp.array(test_dataset['input_ids'])
att_mask=jnp.array(test_dataset['attention_mask'])

outputs=model(input_id,att_mask)

final_output=jnp.argmax(outputs,axis=-1)

logger.info(f"the predction of the test dataset : {final_output[:30]}")
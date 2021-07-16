import jax
print(jax.local_device_count())
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.training.common_utils import get_metrics,onehot,shard,shard_prng_key

from typing import Any, Optional, Tuple

from transformers import (
    GPT2Config)

import transformers
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2",pad_token='<|endoftext|>') 
from datasets import load_dataset,load_metric

from model_file import FlaxGPT2ForMultipleChoice

import logging 

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

seed=0
total_batch_size=32

model = FlaxGPT2ForMultipleChoice.from_pretrained("flax-community/gpt2-Cosmos",input_shape=(1,4,1))

def glue_train_data_loader(rng,dataset,batch_size):
    steps_per_epoch=len_test_dataset//batch_size
    perms=jax.random.permutation(rng,len(dataset))
    perms=perms[:steps_per_epoch*batch_size]
    perms=perms.reshape((steps_per_epoch,batch_size))
    for perm in perms:
      batch=dataset[perm]
      batch={k:jnp.array(v) for k,v in batch.items()}
      batch=shard(batch)
      yield batch

rng=jax.random.PRNGKey(seed)
dropout_rngs=jax.random.split(rng,jax.local_device_count())

input_id=jnp.array(test_dataset['input_ids'])
att_mask=jnp.array(test_dataset['attention_mask'])

restored_output=[]
rng, input_rng = jax.random.split(rng)

for idx,batch in enumerate(glue_train_data_loader(input_rng, test_dataset, total_batch_size)):
    outputs=model(batch['input_ids'],batch['attention_mask'])
    final_output=jnp.argmax(outputs,axis=-1)
    restored_output.append(final_output)

#outputs=model(input_id,att_mask)
#final_output=jnp.argmax(outputs,axis=-1)

logger.info(f"the predction of the test dataset : {restored_output[:30]}")
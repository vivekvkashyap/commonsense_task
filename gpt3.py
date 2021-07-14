import jax
print(jax.local_device_count())
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.training.common_utils import get_metrics,onehot,shard,shard_prng_key
from flax.training import train_state
from flax.metrics.tensorboard import SummaryWriter
from flax.training import checkpoints


import logging
import optax
import math

from pathlib import Path
from typing import Callable
from itertools import chain
from flax.metrics import tensorboard

from datasets import load_dataset,load_metric
from transformers import GPT2Config,GPT2Tokenizer

from model_file  import FlaxGPT2ForMultipleChoice

#logger = logging.getLogger()
#logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

def main():


  tokenizer=GPT2Tokenizer.from_pretrained('gpt2',pad_token='<|endoftext|>')

  dataset=load_dataset('cosmos_qa')

  def preprocess(example):
    example['context&question']=example['context']+example['question']
    example['first_sentence']=[example['context&question'],example['context&question'],example['context&question'],example['context&question']]
    example['second_sentence']=example['answer0'],example['answer1'],example['answer2'],example['answer3']
    return example

  train_dataset=dataset['train'].map(preprocess)
  validation_dataset=dataset['validation'].map(preprocess)
  test_dataset=dataset['test'].map(preprocess)

  #Remove after experiment
  len_train_dataset=64
  len_validation_dataset=64
  len_test_dataset=64

  train_dataset=train_dataset.select(range(len_train_dataset))
  test_dataset=test_dataset.select(range(len_validation_dataset))
  validation_dataset=validation_dataset.select(range(len_test_dataset))

  #remove_cols=train_dataset.column_names

  def tokenize(examples):
    a=tokenizer(examples['first_sentence'],examples['second_sentence'],padding='max_length',truncation=True,max_length=256,return_tensors='jax')
    a['labels']=examples['label']
    return a

  train_dataset=train_dataset.map(tokenize)
  validation_dataset=validation_dataset.map(tokenize)
  test_dataset=test_dataset.map(tokenize)

  remov_col=['id', 'context', 'question', 'answer0', 'answer1', 'answer2', 'answer3', 'labels', 'context&question', 'first_sentence', 'second_sentence']

  train_dataset=train_dataset.remove_columns(remov_col)
  validation_dataset=validation_dataset.remove_columns(remov_col)
  test_dataset=test_dataset.remove_columns(remov_col)

  per_device_batch_size=4
  seed=0
  num_train_epochs=1
  learning_rate=2e-5
  

  total_batch_size = per_device_batch_size * jax.local_device_count()
  print('The overall batch size (both for training and eval) is', total_batch_size)
  num_train_steps = len(train_dataset) // total_batch_size * num_train_epochs
  num_validation_steps=len(validation_dataset)//total_batch_size*num_train_epochs

  learning_rate_function = optax.linear_schedule(init_value=learning_rate, end_value=0, transition_steps=num_train_steps)

  class TrainState(train_state.TrainState):
    logits_function:Callable=flax.struct.field(pytree_node=False)
    loss_function:Callable=flax.struct.field(pytree_node=False)

  def adamw(weight_decay):
    return optax.adamw(learning_rate=learning_rate_function,b1=0.9,b2=0.99,eps=1e-6,weight_decay=weight_decay)

  decay_path=lambda p:not any(x in p for x in ['bias','LayerNorm.weight'])

  def traverse(function):
    def mask(data):
      flat=flax.traverse_util.flatten_dict(data)
      return flax.traverse_util.unflatten_dict({k:function(k,v) for k,v in flat.items()})
    return mask

  gradient_transformation=optax.chain(
      optax.masked(adamw(0.0),mask=traverse(lambda path,_:decay_path(path))),
      optax.masked(adamw(0.01),mask=traverse(lambda path,_:not decay_path(path))))

  def loss_function(logits,labels):
    logits=flax.linen.log_softmax(logits)
    xentropy=optax.softmax_cross_entropy(logits,onehot(labels,num_classes=4))
    return jnp.mean(xentropy)

  def eval_function(logits):
    return logits.argmax(-1)

  model = FlaxGPT2ForMultipleChoice.from_pretrained('gpt2',input_shape=(1,4,1))

  state=TrainState.create(apply_fn=model.__call__,
                          params=model.params,
                          tx=gradient_transformation,
                          logits_function=eval_function,
                          loss_function=loss_function)
                      
  def train_step(state,batch,dropout_rng):
    targets=batch.pop("label")
    dropout_rng,new_dropout_rng=jax.random.split(dropout_rng)
    def loss_function(params):
      logits=state.apply_fn(**batch,params=params,dropout_rng=dropout_rng,train=True)[0]
      loss=state.loss_function(logits,targets)
      return loss

    grad_function=jax.value_and_grad(loss_function)
    loss,grad=grad_function(state.params)
    grad=jax.lax.pmean(grad,"batch")
    new_state=state.apply_gradients(grads=grad)
    #Added.
    logits=new_state.apply_fn(**batch,params=new_state.params,dropout_rng=dropout_rng,train=True)[0]
    accuracy=jnp.equal(jnp.argmax(logits,axis=-1),targets)
    metrics=jax.lax.pmean({"loss":loss,"learning_rate":learning_rate_function(state.step),'accuracy':accuracy},axis_name="batch")
    return new_state,metrics,new_dropout_rng

  parallel_train_step = jax.pmap(train_step, axis_name="batch", donate_argnums=(0,))

  def eval_step(state, batch):
      
      logits = state.apply_fn(**batch, params=state.params, train=False)
      return state.logits_function(logits)

  parallel_eval_step = jax.pmap(eval_step, axis_name="batch")

  def glue_train_data_loader(rng,dataset,batch_size):
    steps_per_epoch=len_train_dataset//batch_size
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

  def glue_eval_data_loader(dataset, batch_size):
      for i in range(len_validation_dataset // batch_size):
          batch = dataset[i * batch_size : (i + 1) * batch_size]
          batch = {k: jnp.array(v) for k, v in batch.items()}
          batch = shard(batch)

          yield batch

  state = flax.jax_utils.replicate(state)
  #metrics_list = list_metrics()

  actual_task = "mnli"
  metric = load_metric('glue', "mnli")
  actual_taskmetric = load_metric('glue', actual_task)

  workdir='./results_tensorboard'
  summary_writer = tensorboard.SummaryWriter(workdir)
  #summary_writer.hparams(dict(GPT2Config()))

  logger.info(f"***** Running training *****")
  logger.info(f"  Num examples = {len_train_dataset}")
  logger.info(f"  Num Epochs = {1}")
  logger.info(f"  Instantaneous batch size per device = {per_device_batch_size}")
  logger.info(f"  Total train batch size (w. parallel & distributed) = {total_batch_size}")
  logger.info(f"  Total optimization steps = {num_train_steps}")

  for i, epoch in enumerate(tqdm(range(1, 2), desc=f"Epoch ...", position=0, leave=True)):
    rng, input_rng = jax.random.split(rng)
    train_acc_metrics=[]
    train_loss_metrics=[]
    eval_acc_metrics=[]
    eval_loss_metrics=[]
    # train
    with tqdm(total=len_train_dataset // total_batch_size, desc="Training...", leave=False) as progress_bar_train:
      for idx,batch in enumerate(glue_train_data_loader(input_rng, train_dataset, total_batch_size)):
        state, train_metric, dropout_rngs = parallel_train_step(state, batch, dropout_rngs)
        train_acc_metrics.append(jax.device_get(train_metric['accuracy']).mean().item())
        train_loss_metrics.append(flax.jax_utils.unreplicate(eval_metric)['loss'].item())
        if idx%5==0:
          summary_writer.scalar('train_loss',flax.jax_utils.unreplicate(train_loss_metric)['loss'].item(),idx)
          summary_writer.scalar('train_accuracy', jax.device_get(train_acc_metrics['accuracy']).mean().item(),idx)
        progress_bar_train.update(1)

    # evaluate
    with tqdm(total=len_validation_dataset // total_batch_size, desc="Evaluating...", leave=False) as progress_bar_eval:
      for idx,batch in enumerate(glue_eval_data_loader(validation_dataset, total_batch_size)):
          labels,predictions,eval_metric=parallel_eval_step(state, batch)
          eval_acc_metrics.append(jax.device_get(eval_acc_metric['accuracy']).mean().item())
          eval_loss_metrics.append(flax.jax_utils.unreplicate(eval_loss_metric)['loss'].item())
          eval_metrics.append(eval_metric)
          progress_bar_eval.update(1)
          if idx%5==0:
            print('eval_step_loss{idx}:',flax.jax_utils.unreplicate(eval_metric)['loss'].item(),
                  'eval_step_acc{idx}':jax.device_get(eval_metric['accuracy']).mean().item())
            summary_writer.scalar('eval_loss',flax.jax_utils.unreplicate(eval_loss_metric)['loss'].item(),idx)
            summary_writer.scalar('eval_accuracy', jax.device_get(eval_acc_metrics['accuracy']).mean().item(),idx)
            #correct
    print('Epoch {epoch} done')
    print('Train loss:',jax.device_get(jnp.array(train_loss_metrics)).mean().item(),
          'Train accuracy:',jax.device_get(jnp.array(train_acc_metrics)).mean().item())
    print('Eval loss:',jax.device_get(jnp.array(eval_loss_metrics)).mean().item(),
          'Eval accuracy:',jax.device_get(jnp.array(eval_acc_metrics)).mean().item())
  summary_writer.flush()

if __name__ == "__main__":
    main()
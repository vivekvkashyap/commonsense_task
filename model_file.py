import jax
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, unfreeze

from typing import Any, Optional, Tuple

from transformers import (
    GPT2Config,
    FlaxGPT2Model)

from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.models.gpt2.modeling_flax_gpt2  import  FlaxGPT2BlockCollection
from transformers.modeling_flax_outputs import FlaxBaseModelOutput
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2Module

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2",pad_token='<|endoftext|>') 

GPT2_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.FlaxPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)
    This model is also a Flax Linen `flax.nn.Module
    <https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html>`__ subclass. Use it as a regular Flax
    Module and refer to the Flax documentation for all matter related to general usage and behavior.
    Finally, this model supports inherent JAX features such as:
    - `Just-In-Time (JIT) compilation <https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit>`__
    - `Automatic Differentiation <https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation>`__
    - `Vectorization <https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap>`__
    - `Parallelization <https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap>`__
    Parameters:
        config (:class:`~transformers.GPT2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.FlaxPreTrainedModel.from_pretrained` method to load the
            model weights.
"""
GPT2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`numpy.ndarray` of shape :obj:`(batch_size,input_ids_length)`):
            :obj:`input_ids_length` = ``sequence_length``. Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`~transformers.GPT2Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`numpy.ndarray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        position_ids (:obj:`numpy.ndarray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.
        past_key_values (:obj:`Dict[str, np.ndarray]`, `optional`, returned by ``init_cache`` or when passing previous ``past_key_values``):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape `[batch_size, max_length]`.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

class FlaxGPT2PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
        self,
        config: GPT2Config,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)["params"]

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (:obj:`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (:obj:`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict
       

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxGPT2Attention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs

class FlaxGPT2Module(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_dim = self.config.hidden_size

        self.wte = nn.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.wpe = nn.Embed(
            self.config.max_position_embeddings,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.h = FlaxGPT2BlockCollection(self.config, dtype=self.dtype)
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        input_embeds = self.wte(input_ids.astype("i4"))
        position_embeds = self.wpe(position_ids.astype("i4"))
        

        hidden_states = input_embeds + position_embeds
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        outputs = self.h(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)
        if not return_dict:
            return (hidden_states,) + outputs[1:]

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,)

class FlaxGPT2Model(FlaxGPT2PreTrainedModel):
    module_class = FlaxGPT2Module

class FlaxGPT2ForMultipleChoiceModule(nn.Module):
  config:GPT2Config
  dtype: jnp.dtype = jnp.float32
  def setup(self):
    self.gpt2 = FlaxGPT2Module(config=self.config, dtype=self.dtype)
    self.dropout = nn.Dropout(rate=0.2)
    self.classifier = nn.Dense(4, dtype=self.dtype)

  def __call__(self,input_ids,attention_mask,position_ids,return_dict=True,deterministic=True,*args):
    batch_size = input_ids.shape[0]
    rng=jax.random.PRNGKey(0)
    _, dropout_rng = jax.random.split(rng)
    input_ids=input_ids.reshape(4*batch_size,-1)
    position_ids=position_ids.reshape(4*batch_size,-1)
    attention_mask=attention_mask.reshape(4*batch_size,-1) 

    outputs=self.gpt2(input_ids, attention_mask,position_ids,return_dict=return_dict)
    

    hidden_states = outputs[0]

    
    hidden_states= jnp.mean(hidden_states, axis=1)

    
    
    hidden_states=hidden_states.reshape(batch_size,-1)         #(32,8,768)->(32,8*768)

    dropout_output = self.dropout(hidden_states,deterministic=deterministic,rng=dropout_rng)

    

    logits = self.classifier(dropout_output)
    reshaped_logits = logits.reshape(-1, 4)   
                  #(32,4)
    if not return_dict:
      return (reshaped_logits,) + outputs[2:]
    return reshaped_logits

class FlaxGPT2ForMultipleChoice(FlaxGPT2Model):
    module_class = FlaxGPT2ForMultipleChoiceModule


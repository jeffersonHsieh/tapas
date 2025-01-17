# coding=utf-8
# Copyright 2019 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""The main BERT model and related functions."""

import collections
import copy
import json
import math
import re
from absl import logging
import numpy as np
import six
from tapas.models import segmented_tensor
import tensorflow.compat.v1 as tf
import tf_slim


class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02,
               softmax_temperature=1.0):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
      softmax_temperature: The temperature for the attention softmax.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    self.softmax_temperature = softmax_temperature

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

  def to_json_file(self, json_file):
    """Serializes this instance to a JSON file."""
    with tf.io.gfile.GFile(json_file, "w") as writer:
      writer.write(self.to_json_string())


class BertModel(object):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               attention_mask=None,
               token_weights=None,
               custom_attention_layer=None,
               custom_transformer_layer=None,
               token_type_ids=None,
               extra_embeddings=None,
               use_position_embeddings=True,
               reset_position_index_per_cell=False,
               proj_value_length=None,
               scope=None,
               init_checkpoint=None,#MODIFIED by Chia-Chun
               load_custom_segment_vocab_size = False #MODIFIED by Chia-Chun
               ):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      attention_mask: (optional) float32 Tensor of shape
        [batch_size, seq_length, seq_length].
      token_weights: (optional) float32 Tensor of shape
        [batch_size, seq_length] in [0,1].
      custom_attention_layer: (optional) function with the same signature as
        `attention_layer` in order to replace it for sparse alternatives.
      custom_transformer_layer: (optional) function with the same signature as
        `transformer_model` in order to replace for sparse alternatives.
      token_type_ids: (optional) nested structure of int32 Tensors of shape
        [batch_size, seq_length].
      extra_embeddings: (optional) float32 Tensor of shape [batch_size, seq_len,
        embedding_dim]. Additional embeddings concatenated with all the other
        embeddings.
      use_position_embeddings: (optional) bool. Whether to use position
        embeddings.
      reset_position_index_per_cell: bool. Whether to restart position index
        when a new cell starts.
      proj_value_length: (optional) int. If set, used to down-project key
        and value tensors (following https://arxiv.org/pdf/2006.04768.pdf).
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
    if token_weights is not None:
      input_mask = token_weights * tf.cast(input_mask, dtype=tf.float32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings")

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=use_position_embeddings,
            reset_position_index_per_cell=reset_position_index_per_cell,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            extra_embeddings=extra_embeddings,
            dropout_prob=config.hidden_dropout_prob,
            init_checkpoint=init_checkpoint, #MODIFIED by Chia-Chun
            load_custom_segment_vocab_size = load_custom_segment_vocab_size #MODIFIED by Chia-Chun
            )

      # import pdb;pdb.set_trace()
      # can get the created variables in pdb with: interact;
      # with tf.variable_scope('embeddings', reuse=True):
      #   tf.get_variable("token_type_embeddings_0") 
      
      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        if attention_mask is None:
          attention_mask = create_attention_mask_from_input_mask(
              input_ids, input_mask)

        transformer_layer = custom_transformer_layer or transformer_model

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers, self.all_attention_probs = transformer_layer(
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            input_mask=input_mask,
            custom_attention_layer=custom_attention_layer,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True,
            do_return_attention_probs=True,
            softmax_temperature=config.softmax_temperature,
            proj_value_length=proj_value_length)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_all_attention_probs(self):
    return self.all_attention_probs

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint, scope=None):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    # import pdb;pdb.set_trace()
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var
    # import pdb;pdb.set_trace()

  init_vars = tf.train.list_variables(init_checkpoint)

  # Map from names in the checkpoint to names in the graph
  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (short_name, var) = (x[0], x[1])
    # Name in the graph
    name = f"{scope}/{short_name}" if scope else short_name
    if name not in name_to_variable:
      continue
    assignment_map[short_name] = name_to_variable[name]
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1
    # about the ":0" in variable names: first output of the node https://stackoverflow.com/questions/40925652/in-tensorflow-whats-the-meaning-of-0-in-a-variables-name

  return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, rate=dropout_prob)
  return output


def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf_slim.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings"):
  """Looks up words embeddings for id tensor.

  Args:
    input_ids: int32 Tensor of an arbitrary shape containing word ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialization range.
    word_embedding_name: string. Name of the embedding table.

  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))

  input_shape = get_shape_list(input_ids)
  output = tf.nn.embedding_lookup(embedding_table, input_ids)
  output = tf.reshape(output, input_shape + [embedding_size])
  return (output, embedding_table)


def _get_absolute_position_embeddings(full_position_embeddings, seq_length,
                                      width, num_dims):
  """Compute absolute position embeddings."""
  # Since the position embedding table is a learned variable, we create it
  # using a (long) sequence length `max_position_embeddings`. The actual
  # sequence length might be shorter than this, for faster training of
  # tasks that do not have long sequences.
  #
  # So `full_position_embeddings` is effectively an embedding table
  # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
  # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
  # perform a slice.
  position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                 [seq_length, -1])

  # Only the last two dimensions are relevant (`seq_length` and `width`), so
  # we broadcast among the first dimensions, which is typically just
  # the batch size.
  position_broadcast_shape = []
  for _ in range(num_dims - 2):
    position_broadcast_shape.append(1)
  position_broadcast_shape.extend([seq_length, width])
  position_embeddings = tf.reshape(position_embeddings,
                                   position_broadcast_shape)
  return position_embeddings


def _get_relative_position_embeddings(
    full_position_embeddings,
    token_type_ids,
    token_type_vocab_size,
    seq_length,
    batch_size,
    max_position_embeddings,
):
  """Create position embeddings that restart at every cell."""
  col_index = segmented_tensor.IndexMap(
      token_type_ids[1], token_type_vocab_size[1], batch_dims=1)
  row_index = segmented_tensor.IndexMap(
      token_type_ids[2], token_type_vocab_size[2], batch_dims=1)
  full_index = segmented_tensor.ProductIndexMap(col_index, row_index)
  position = tf.expand_dims(tf.range(seq_length), axis=0)
  logging.info("position: %s", position)
  batched_position = tf.repeat(position, repeats=batch_size, axis=0)
  logging.info("batched_position: %s", batched_position)
  logging.info("token_type_ids: %s", token_type_ids[1])
  first_position_per_segment = segmented_tensor.reduce_min(
      batched_position, full_index)[0]
  first_position = segmented_tensor.gather(first_position_per_segment,
                                           full_index)
  position_embeddings = tf.nn.embedding_lookup(
      full_position_embeddings,
      tf.math.minimum(max_position_embeddings - 1, position - first_position))
  return position_embeddings


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=None,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            reset_position_index_per_cell=False,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            extra_embeddings=None,
                            dropout_prob=0.1,
                            init_checkpoint=None, #MODIFIED by Chia-Chun
                            load_custom_segment_vocab_size=False #MODIFIED by Chia-Chun
                            ):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) nested structure of int32 Tensors of shape
      [batch_size, seq_length]. Must be specified if `use_token_type` is True.
    token_type_vocab_size: nested structure of ints. The vocabulary size of
      `token_type_ids`. Must match the structure of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    reset_position_index_per_cell: bool. Whether to restart position index when
      a new cell starts.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    extra_embeddings: (optional) float32 Tensor of shape [batch_size,
      seq_length, embedding_dim]. Additional embeddings concatenated with all
      the other embeddings.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")

    tf.nest.assert_same_structure(token_type_ids, token_type_vocab_size)
    token_type_ids = tf.nest.flatten(token_type_ids)
    token_type_vocab_size = tf.nest.flatten(token_type_vocab_size)

    for i, (type_ids, type_vocab_size) in enumerate(
        zip(token_type_ids, token_type_vocab_size)):

      if i==0 and init_checkpoint is not None and load_custom_segment_vocab_size:
        import numpy as np
        seg_embs = tf.train.load_variable(init_checkpoint,'bert/embeddings/token_type_embeddings_0')
        new_seg_embs = np.array(seg_embs)
        rng = np.random.default_rng(0)
        new_seg_embs = np.vstack([seg_embs,rng.normal(0,0.02,[type_vocab_size-seg_embs.shape[0],seg_embs.shape[1]])]).astype(np.float32)
        init_new_seg_embs = tf.constant(new_seg_embs)

        # this method just uses a new name for the variable -- since it's not in checkpoint, won't get overwritten
        token_type_table = tf.get_variable(
            name=token_type_embedding_name,#"%s_%d" % (token_type_embedding_name, i),
            initializer=init_new_seg_embs)
        
        # this method requires you to pop token_type_embedding_0 from the assignment map later to avoid loading from checkpoint and causing shape mismatch
        # token_type_table = tf.get_variable(
        #     name="%s_%d" % (token_type_embedding_name, i),
        #     initializer=init_new_seg_embs)
      else:
        token_type_table = tf.get_variable(
            name="%s_%d" % (token_type_embedding_name, i),
            shape=[type_vocab_size, width],
            initializer=create_initializer(initializer_range)) #creates a tensorflow variable
        
      # This vocab will be small so we always do one-hot here, since it is
      # always faster for a small vocabulary.
      
      # debug inspection
      # if i==0:
      #   token_type_table = tf.Print(token_type_table,[token_type_table[:,:5]],summarize=200)
        # type_ids = tf.Print(type_ids,[type_ids], summarize=200)

      flat_token_type_ids = tf.reshape(type_ids, [-1])
      # if i==0:
      #   import pdb;pdb.set_trace()
      #   type_vocab_size = tf.unique(flat_token_type_ids)[0].shape[0]
      #   print(type_vocab_size)
      one_hot_ids = tf.one_hot(flat_token_type_ids, depth=type_vocab_size)
      # if i==0:
      #   # import pdb;pdb.set_trace()
      #   one_hot_ids = tf.Print(one_hot_ids,[one_hot_ids], summarize=50*3)
      token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
      # if i==0:
      #   token_type_embeddings = tf.Print(token_type_embeddings,[token_type_embeddings], summarize=200)
      token_type_embeddings = tf.reshape(token_type_embeddings,
                                         [batch_size, seq_length, width])
      output += token_type_embeddings

  if use_position_embeddings:
    full_position_embeddings = tf.get_variable(
        name=position_embedding_name,
        shape=[max_position_embeddings, width],
        initializer=create_initializer(initializer_range))
    if not reset_position_index_per_cell:
      assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
      with tf.control_dependencies([assert_op]):
        num_dims = len(output.shape.as_list())
        position_embeddings = _get_absolute_position_embeddings(
            full_position_embeddings,
            seq_length=seq_length,
            width=width,
            num_dims=num_dims,
        )
    else:
      position_embeddings = _get_relative_position_embeddings(
          full_position_embeddings,
          token_type_ids,
          token_type_vocab_size,
          seq_length,
          batch_size,
          max_position_embeddings,
      )
    output += position_embeddings

  if extra_embeddings is not None:
    flat_extra_embeddings = tf.reshape(extra_embeddings,
                                       [batch_size * seq_length, -1])
    flat_extra_embeddings = tf.layers.dense(
        flat_extra_embeddings,
        width,
        kernel_initializer=create_initializer(initializer_range))
    output += tf.reshape(flat_extra_embeddings, [batch_size, seq_length, width])

  output = layer_norm_and_dropout(output, dropout_prob)
  return output


def create_attention_mask_from_input_mask(
    from_tensor,
    to_mask,
):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask


def dense_layer_3d(input_tensor,
                   num_attention_heads,
                   size_per_head,
                   initializer,
                   activation,
                   name=None):
  """A dense layer with 3D kernel.

  Args:
    input_tensor: float Tensor of shape [batch, num_attention_heads,
      size_per_head].
    num_attention_heads: Number of attention heads.
    size_per_head: The size per attention head.
    initializer: Kernel initializer.
    activation: Actication function.
    name: The name scope of this layer.

  Returns:
    float logits Tensor.
  """

  last_dim = get_shape_list(input_tensor)[-1]

  with tf.variable_scope(name):
    w = tf.get_variable(
        name="kernel",
        shape=[last_dim, num_attention_heads * size_per_head],
        initializer=initializer)
    w = tf.reshape(w, [last_dim, num_attention_heads, size_per_head])
    b = tf.get_variable(
        name="bias",
        shape=[num_attention_heads * size_per_head],
        initializer=tf.zeros_initializer)
    b = tf.reshape(b, [num_attention_heads, size_per_head])
    ret = tf.einsum("abc,cde->abde", input_tensor, w)
    ret += b
    if activation is not None:
      return activation(ret)
    else:
      return ret


def dense_layer_3d_proj(input_tensor,
                        hidden_size,
                        num_attention_heads,
                        head_size,
                        initializer,
                        activation,
                        name=None):
  """A dense layer with 3D kernel for projection.

  Args:
    input_tensor: float Tensor of shape [batch,from_seq_length,
      num_attention_heads, size_per_head].
    hidden_size: The size of hidden layer.
    num_attention_heads: The size of output dimension.
    head_size: The size of head.
    initializer: Kernel initializer.
    activation: Actication function.
    name: The name scope of this layer.

  Returns:
    float logits Tensor.
  """
  head_size = hidden_size // num_attention_heads
  with tf.variable_scope(name):
    w = tf.get_variable(
        name="kernel",
        shape=[hidden_size, hidden_size],
        initializer=initializer)
    w = tf.reshape(w, [num_attention_heads, head_size, hidden_size])
    b = tf.get_variable(
        name="bias", shape=[hidden_size], initializer=tf.zeros_initializer)

  ret = tf.einsum("BFNH,NHD->BFD", input_tensor, w)
  ret += b
  if activation is not None:
    return activation(ret)
  else:
    return ret


def dense_layer_2d(input_tensor,
                   output_size,
                   initializer,
                   activation,
                   name=None):
  """A dense layer with 2D kernel.

  Args:
    input_tensor: Float tensor with rank 3.
    output_size: The size of output dimension.
    initializer: Kernel initializer.
    activation: Actication function.
    name: The name scope of this layer.

  Returns:
    float logits Tensor.
  """
  last_dim = get_shape_list(input_tensor)[-1]
  with tf.variable_scope(name):
    w = tf.get_variable(
        name="kernel", shape=[last_dim, output_size], initializer=initializer)
    b = tf.get_variable(
        name="bias", shape=[output_size], initializer=tf.zeros_initializer)

  ret = tf.einsum("abc,cd->abd", input_tensor, w)
  ret += b
  if activation is not None:
    return activation(ret)
  else:
    return ret


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    input_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    softmax_temperature=1.0,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    to_proj_length=None):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with tf.einsum as follows:
    Input_tensor: [BFD]
    Wq, Wk, Wv: [DNH]
    Q:[BFNH] = einsum('BFD,DNH->BFNH', Input_tensor, Wq)
    K:[BTNH] = einsum('BTD,DNH->BTNH', Input_tensor, Wk)
    V:[BTNH] = einsum('BTD,DNH->BTNH', Input_tensor, Wv)
    attention_scores:[BNFT] = einsum('BFNH,BTNH>BNFT', Q, K) / sqrt(H)
    attention_probs:[BNFT] = softmax(attention_scores)
    context_layer:[BFNH] = einsum('BNFT,BTNH->BFNH', attention_probs, V)
    Wout:[DNH]
    Output:[BFD] = einsum('BFNH,DNH>BFD', context_layer, Wout)

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    input_mask: Only required when using to_proj_length.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    softmax_temperature: The temperature for the softmax attention.
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.
    to_proj_length: (Optional) Int. Down-project keys and values to this length.

  Returns:
    float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
      size_per_head].

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  # `query_layer` = [B, F, N, H]
  query_layer = dense_layer_3d(from_tensor, num_attention_heads, size_per_head,
                               create_initializer(initializer_range), query_act,
                               "query")

  # `key_layer` = [B, T, N, H]
  key_layer = dense_layer_3d(to_tensor, num_attention_heads, size_per_head,
                             create_initializer(initializer_range), key_act,
                             "key")

  # `value_layer` = [B, T, N, H]
  value_layer = dense_layer_3d(to_tensor, num_attention_heads, size_per_head,
                               create_initializer(initializer_range), value_act,
                               "value")

  if to_proj_length is not None:
    # This gives one project matrix per layer (shared by heads and value/key).
    # In the paper they also look into other sharing schemes.
    with tf.variable_scope("proj_seq_length"):
      proj_kernel = tf.get_variable(
          name="kernel",
          shape=[to_seq_length, to_proj_length],
          initializer=create_initializer(initializer_range))

    input_mask = tf.cast(input_mask, tf.float32)
    input_mask4d = tf.reshape(input_mask, (batch_size, to_seq_length, 1, 1))

    key_layer = key_layer * input_mask4d
    # [B, K, N, H]
    key_layer = tf.einsum("BTNH,TK->BKNH", key_layer, proj_kernel)

    value_layer = value_layer * input_mask4d
    # [B, K, N, H]
    value_layer = tf.einsum("BTNH,TK->BKNH", value_layer, proj_kernel)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  attention_scores = tf.einsum(
      "BFNH,BTNH->BNFT", query_layer, key_layer, name="query_key_einsum")

  attention_scores = attention_scores / softmax_temperature
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None and to_proj_length is None:
    # `attention_mask` = [B, 1, F, T] or [B, H, F, T]
    # Caller can pass a rank 3 tensor for a constand mask or rank 4 for per-head
    # head attention mask.
    attention_mask = tf.reshape(
        attention_mask, shape=[batch_size, -1, from_seq_length, to_seq_length])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    attention_mask_float = tf.cast(attention_mask, tf.float32)
    # Please keep this tf.where as it fixes back propagation issues: It removes
    # NaNs when using tf.math.log.
    attention_mask_float = tf.where(attention_mask_float > 0.0,
                                    attention_mask_float,
                                    tf.zeros_like(attention_mask_float))

    adder = tf.math.log(attention_mask_float)
    adder = tf.where(
        tf.is_finite(adder), adder,
        tf.zeros_like(adder, dtype=tf.float32) - 10000.0)

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs_do = dropout(attention_probs, attention_probs_dropout_prob)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.einsum(
      "BNFT,BTNH->BFNH",
      attention_probs_do,
      value_layer,
      name="attention_value_einsum")

  return context_layer, attention_probs


def transformer_model(input_tensor,
                      attention_mask=None,
                      input_mask=None,
                      custom_attention_layer=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      softmax_temperature=1.0,
                      do_return_all_layers=False,
                      do_return_attention_probs=False,
                      proj_value_length=None):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    input_mask: ...
    custom_attention_layer: (optional) function with the same signature as
      `attention_layer` in order to replace it for sparse alternatives.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    softmax_temperature: The temperature for the softmax attention.
    do_return_all_layers: Whether to also return all layers or just the final
      layer.
    do_return_attention_probs: Whether to also return all layers self-attention
      matrix.
    proj_value_length: (optional) int. If set, used to down-project key and
      value tensors (following https://arxiv.org/pdf/2006.04768.pdf).

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  prev_output = input_tensor
  all_layer_outputs = []
  all_attention_probs = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output

      with tf.variable_scope("attention"):
        with tf.variable_scope("self"):
          custom_attention_layer = custom_attention_layer or attention_layer
          attention_output, attention_probs = custom_attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=attention_mask,
              input_mask=input_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              softmax_temperature=softmax_temperature,
              to_proj_length=proj_value_length,
          )

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
          attention_output = dense_layer_3d_proj(
              attention_output, hidden_size,
              num_attention_heads, attention_head_size,
              create_initializer(initializer_range), None, "dense")
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = dense_layer_2d(
            attention_output, intermediate_size,
            create_initializer(initializer_range), intermediate_act_fn, "dense")

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = dense_layer_2d(intermediate_output, hidden_size,
                                      create_initializer(initializer_range),
                                      None, "dense")
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)
        all_attention_probs.append(attention_probs)

  if do_return_all_layers:
    if do_return_attention_probs:
      return all_layer_outputs, all_attention_probs
    return all_layer_outputs
  else:
    return all_layer_outputs[-1]


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

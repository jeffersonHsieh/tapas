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
"""TABLE BERT utility functions."""

from tapas.models.bert import modeling
from tapas.utils import attention_utils

import tensorflow.compat.v1 as tf

_AttentionMode = attention_utils.RestrictAttentionMode


def create_model(
    features,
    mode,
    bert_config,
    restrict_attention_mode=_AttentionMode.FULL,
    restrict_attention_bucket_size=0,
    restrict_attention_header_size=None,
    restrict_attention_row_heads_ratio=0.5,
    restrict_attention_sort_after_projection=True,
    token_weights=None,
    disabled_features=None,
    disable_position_embeddings=False,
    reset_position_index_per_cell=False,
    proj_value_length=None,
):
  """Creates a TABLE BERT model."""
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)

  token_type_features = [
      "segment_ids", "column_ids", "row_ids", "prev_label_ids", "column_ranks",
      "inv_column_ranks", "numeric_relations"
  ]
  token_type_ids = []
  for key in token_type_features:
    if disabled_features is not None and key in disabled_features:
      token_type_ids.append(tf.zeros_like(features[key]))
    else:
      token_type_ids.append(features[key])

  attention_mask = None
  custom_attention_layer = None
  num_row_heads = int(bert_config.num_attention_heads *
                      restrict_attention_row_heads_ratio)
  num_column_heads = bert_config.num_attention_heads - num_row_heads
  if restrict_attention_mode == _AttentionMode.HEADWISE_SAME_COLUMN_OR_ROW:
    attention_mask = attention_utils.compute_headwise_sparse_attention_mask(
        num_row_heads=num_row_heads,
        num_column_heads=num_column_heads,
        bucket_size=restrict_attention_bucket_size,
        header_size=restrict_attention_header_size,
        **features)
  elif restrict_attention_mode == _AttentionMode.SAME_COLUMN_OR_ROW:
    attention_mask = attention_utils.compute_sparse_attention_mask(**features)
  elif restrict_attention_mode == _AttentionMode.HEADWISE_EFFICIENT:
    custom_attention_layer = attention_utils.create_bucketed_attention_layer(
        input_mask=features["input_mask"],
        input_header=tf.math.equal(features["segment_ids"], 0),
        bucket_size=restrict_attention_bucket_size,
        header_size=restrict_attention_header_size,
        sort_after_projection=restrict_attention_sort_after_projection,
        token_type_ids=[(num_row_heads, True, features["row_ids"]),
                        (num_column_heads, False, features["column_ids"])])
  elif restrict_attention_mode == _AttentionMode.FULL:
    pass
  else:
    raise ValueError(f"Unknown attention mode: {restrict_attention_mode}")

  return modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=features["input_ids"],
      input_mask=features["input_mask"],
      attention_mask=attention_mask,
      custom_attention_layer=custom_attention_layer,
      token_weights=token_weights,
      token_type_ids=token_type_ids,
      use_position_embeddings=not disable_position_embeddings,
      reset_position_index_per_cell=reset_position_index_per_cell,
      proj_value_length=proj_value_length,
  )

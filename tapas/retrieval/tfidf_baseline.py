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
"""A simple TF-IDF model for table retrieval."""

from typing import Iterable, List, Text

from absl import app
from absl import flags
from absl import logging
import pandas as pd
from tapas.protos import interaction_pb2
from tapas.retrieval import tfidf_baseline_utils
from tapas.scripts import prediction_utils

#---debug starts---
from collections import defaultdict
import json
from pathlib import Path
import itertools
import time

#---debug ends---

FLAGS = flags.FLAGS

flags.DEFINE_list("interaction_files", None,
                  "Interaction protos in tfrecord format.")

flags.DEFINE_string("table_file", None, "Table protos in tfrecord format.")

flags.DEFINE_integer("max_table_rank", 50, "Max number of tables to retrieve.")

flags.DEFINE_integer("min_term_rank", 100,
                     "Min term frequency rank to consider.")

flags.DEFINE_boolean("drop_term_frequency", True,
                     "If True, ignore term frequency term.")
flags.DEFINE_string("error_log_dir",None,"path to log error ids")
#useless because queries in the dataset are all uncased
flags.DEFINE_boolean("cased",False,"does not lower string in corpus & query(?) if cased")
flags.DEFINE_boolean("oracle_abbv",False,"expand abbv in query using ref table abbv list")
flags.DEFINE_string("tid_abbv_mapping_file",None,"path to tid2abbv mapping")
flags.DEFINE_integer("sleep_after_hparam", None, "secs to sleep after printing hparams")

def _print(message):
  logging.info(message)
  print(message)


def evaluate(index, max_table_rank,
             thresholds,
             interactions,
             rows,
             #-------custom args-------
             split,
             hparam,
             log_dir,
             cased=False,
             oracle_abbv=False,
             tid_to_abbv=None,
             filter_stop_words=False
            #-------custom args end-------
             ):
  """Evaluates index against interactions."""
  ranks = []
  
  #---debug starts---
  out = []
  #---debug ends---

  for nr, interaction in enumerate(interactions):
    for question in interaction.questions:
      # query expansion (original version)
    # TODO: remove this------------
      local_tid_to_abbv=None
      if oracle_abbv:
        local_tid_to_abbv_cased = tid_to_abbv[interaction.table.table_id]
        local_tid_to_abbv = {}
        for abbv,expansion in local_tid_to_abbv_cased.items():
          if abbv in local_tid_to_abbv:
            raise ValueError("duplicate abbv in oracle_abbv")
          local_tid_to_abbv[abbv.lower()]=expansion
    # -----------------------------

      scored_hits = index.retrieve(
        question.original_text,
        cased=cased,
        use_oracle_abbv=oracle_abbv,
        oracle_abbv_map=local_tid_to_abbv,
        filter_stop_words=filter_stop_words
        )
      reference_table_id = interaction.table.table_id

    #---debug starts---
      o = {}
      o['qid'] = interaction.id #question.id
      o['scores'] = scored_hits[:max_table_rank]
      o['ref_table_id'] = reference_table_id
    #---debug ends---

      for rank, (table_id, _) in enumerate(scored_hits[:max_table_rank]):
        if table_id == reference_table_id:
          ranks.append(rank)
        #---debug starts---
          o['rank'] = rank
        #---debug ends---
          break
      
    #---debug starts---
      if 'rank' not in o:
        o['rank'] = 100000
      out.append(o)
    #---debug ends---
    if nr % (len(interactions) // 10) == 0:
      _print(f"Processed {nr:5d} / {len(interactions):5d}.")

  def precision_at_th(threshold):
    return sum(1 for rank in ranks if rank < threshold) / len(interactions)

  values = [f"{precision_at_th(threshold):.4}" for threshold in thresholds]
  rows.append(values)

#---debug starts---
  if log_dir:
    with open(log_dir/f'bm25_{split}_{hparam}_results.jsonl','w') as f:
      for o in out:
        json.dump(o,f)
        f.write('\n')
    _print(f'error logged to {str(log_dir)}/bm25_{split}_{hparam}_results.jsonl')
#---debug ends---

def create_index(tables,
                 title_multiplicator, use_bm25):
  if use_bm25:
    return tfidf_baseline_utils.create_bm25_index(
        tables,
        title_multiplicator=title_multiplicator,
    )
  return tfidf_baseline_utils.create_inverted_index(
      tables=tables,
      min_rank=FLAGS.min_term_rank,
      drop_term_frequency=FLAGS.drop_term_frequency,
      title_multiplicator=title_multiplicator,
  )

# --------------- custom starts -----------------
def create_custom_index(
  tables,title_multiplicator,
  weight_header, 
  weight_sec_title,
  weight_caption,
  weight_content,
  weight_abbv,
  cased = False,
  filter_abbv=False,
  use_bm25=True
  ):
  if use_bm25:
    return tfidf_baseline_utils.create_uneven_bm25_index(
        tables=tables,
        title_multiplicator=title_multiplicator,
        weight_sec_title = weight_sec_title,
        weight_caption = weight_caption,
        weight_header=weight_header,
        weight_content=weight_content,
        weight_abbv=weight_abbv,
        cased = cased,
        filter_abbv=filter_abbv
    )
  else:
    return tfidf_baseline_utils.create_uneven_inverted_index(
        tables=tables,
        title_multiplicator=title_multiplicator,
        weight_sec_title = weight_sec_title,
        weight_caption = weight_caption,
        weight_header=weight_header,
        weight_content=weight_content,
        weight_abbv=weight_abbv,
        cased = cased,
        filter_abbv=filter_abbv,
        min_rank=FLAGS.min_term_rank,
        drop_term_frequency=FLAGS.drop_term_frequency
    )



# --------------- custom ends -----------------

def get_hparams():
  hparams = []
  # debug
  #for multiplier in [1, 2]:
  #  hparams.append({"multiplier": multiplier, "use_bm25": False})
  for multiplier in [15]: #[10, 15]:
  # debug ends
    hparams.append({"multiplier": multiplier, "use_bm25": True})
  return hparams

# --------------- custom starts -----------------
def get_exp_hparams():
  _print("using custom hyper params for header and table")
  params = {
      "w_c": [1], # content multiplier
      "w_h": [15], # header multiplier
      "multiplier": [15], #title multiplier
      "w_cap": [0], # caption multiplier
      "w_sec": [15,0], # section title multiplier
      "use_bm25":[True],
      "abbv":[0],
      "filter_abbv":[False],
      "filter_stop_words":[True]
      
  }
  hparams = []
  for xs in itertools.product(*params.values()):
    if not any(xs[:-1]):
      print('all zeros!')
      continue
    hparams.append(dict(zip(params.keys(), xs)))
  
  print(hparams)
  if FLAGS.sleep_after_hparam:
    _print(f"sleeping for {FLAGS.sleep_after_hparam}s")
    time.sleep(FLAGS.sleep_after_hparam)
  return hparams
# --------------- custom ends -----------------

def main(_):
  tid_to_abbv = None
  if FLAGS.oracle_abbv:
    if not FLAGS.tid_abbv_mapping_file:
      raise ValueError('tid_abbv_mapping_file is required for oracle_abbv')
    with open(FLAGS.tid_abbv_mapping_file) as f:
      tid_to_abbv = json.load(f)
  max_table_rank = FLAGS.max_table_rank
  thresholds = [1,5,10,15,50,100,500, 1000, max_table_rank] #[1, 5, 10, 15, max_table_rank]
  log_dir=None
  if FLAGS.error_log_dir:
    log_dir = Path(FLAGS.error_log_dir)
    log_dir.mkdir(exist_ok=True,parents=True)

  for interaction_file in FLAGS.interaction_files:
    _print(f"Test set: {interaction_file}")
    interactions = list(prediction_utils.iterate_interactions(interaction_file))

    for use_local_index in [False]: #[True, False]:

      rows = []
      row_names = []
# --------------- custom starts -----------------
      for hparams in get_exp_hparams(): #get_hparams():
# --------------- custom ends -------------------
        name = "local" if use_local_index else "global"
        name += "_bm25" if hparams["use_bm25"] else "_tfidf"
# --------------- custom starts -----------------
        name += f'_t1m{hparams["multiplier"]}'
        name += f'_t2m{hparams["w_sec"]}'
        name += f'_t3m{hparams["w_cap"]}'
        name += f'_hm{hparams["w_h"]}'
        name += f'_cm{hparams["w_c"]}'
        name += f'_am{hparams["abbv"]}'
        name += '_filter_stw' if hparams["filter_stop_words"] else ''
        name += "_filter_abbv" if hparams["filter_abbv"] else ''
        if FLAGS.cased:
          name += '_cased'
        if FLAGS.oracle_abbv:
          name+='_oracle_query_exp'
# --------------- custom ends -----------------
        _print(name)
        if use_local_index:
          index = create_index(
              tables=(i.table for i in interactions),
              title_multiplicator=hparams["multiplier"],
              use_bm25=hparams["use_bm25"],
          )
        else:
          # --------------- custom starts -----------------
          index = create_custom_index(
              tables=tfidf_baseline_utils.iterate_tables(FLAGS.table_file),
              title_multiplicator=hparams["multiplier"],
              weight_header=hparams["w_h"],
              weight_sec_title=hparams["w_sec"],
              weight_caption=hparams["w_cap"],
              weight_content=hparams["w_c"],
              weight_abbv=hparams["abbv"],
              cased = FLAGS.cased,
              filter_abbv=hparams["filter_abbv"]
            )
          # index = create_index(
          #     tables=tfidf_baseline_utils.iterate_tables(FLAGS.table_file),
          #     title_multiplicator=hparams["multiplier"],
          #     use_bm25=hparams["use_bm25"],
          # )
          # --------------- custom ends -----------------

        _print("... index created.")
        #-------------added custom args-----------------
        split = interaction_file.split('/')[-1].split('.')[0]


        evaluate(
          index, max_table_rank, thresholds, interactions, rows, 
          split, name, log_dir=log_dir,cased=FLAGS.cased, oracle_abbv=FLAGS.oracle_abbv,
          tid_to_abbv=tid_to_abbv, filter_stop_words=hparams['filter_stop_words'])
        row_names.append(name)

        df = pd.DataFrame(rows, columns=thresholds, index=row_names)
        _print(df.to_string())


if __name__ == "__main__":
  flags.mark_flag_as_required("interaction_files")
  flags.mark_flag_as_required("table_file")
  app.run(main)

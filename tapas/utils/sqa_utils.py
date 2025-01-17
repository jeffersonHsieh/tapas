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
"""Reads interactions from TSV files adds tables and writes to tfrecords."""

import collections
import csv
import os
from typing import Iterable, Mapping, Text, Optional
import json
import time

from absl import logging
from tapas.protos import interaction_pb2
from tapas.utils import file_utils
from tapas.utils import interaction_utils
from tapas.utils import interaction_utils_parser
from tapas.utils import pruning_utils
import tensorflow.compat.v1 as tf


_Mode = interaction_utils_parser.SupervisionMode
_Interactions = Iterable[interaction_pb2.Interaction]


def _read_interactions(input_dir):
  """Reads interactions from TSV files."""
  filenames = [
      fn for fn in file_utils.list_directory(input_dir) if fn.endswith('.tsv') and 'train' not in fn
  ]
  print('temp disabled reading train files in sqa_utils.py:_read_interactions')
  time.sleep(5)
  interaction_dict = {}
  for filename in filenames:
    filepath = os.path.join(input_dir, filename)
    with tf.io.gfile.GFile(filepath, 'r') as file_handle:
      try:
        interactions = interaction_utils.read_from_tsv_file(file_handle)
        interaction_dict[filename] = interactions
      except KeyError as ke:
        logging.error("Can't read interactions from file: %s (%s)", filepath,
                      ke)
  return interaction_dict


def _add_tables(input_dir,
                interaction_dict):
  """Adds table protos to all interactions."""
  table_files = set()
  for interactions in interaction_dict.values():
    for interaction in interactions:
      table_files.add(interaction.table.table_id)

  table_dict = {}
  for index, table_file in enumerate(sorted(table_files)):
    logging.log_every_n(logging.INFO, 'Read %4d / %4d table files', 100, index,
                        len(table_files))
    table_path = os.path.join(input_dir, table_file)
    with tf.io.gfile.GFile(table_path, 'r') as table_handle:
      table = interaction_pb2.Table()
      rows = list(csv.reader(table_handle))
      headers, rows = rows[0], rows[1:]

      for header in headers:
        table.columns.add().text = header

      for row in rows:
        new_row = table.rows.add()
        for cell in row:
          new_row.cells.add().text = cell

      table.table_id = table_file
      table_dict[table_file] = table

  for interactions in interaction_dict.values():
    for interaction in interactions:
      interaction.table.CopyFrom(table_dict[interaction.table.table_id])


def _write_report(
    report_filename,
    supervision_modes,
    counters,
):
  """Creates the report_filename file containing statistics about conversion."""
  contents = []
  contents.append('\t'.join(('Total', 'Valid', 'Failed', 'File')))
  for dataset, counter in counters.items():
    failed = counter.pop('failed', 0)
    valid = counter.pop('valid', 0)
    total = failed + valid
    contents.append('\t'.join(map(str, (total, valid, failed, dataset))))

  # Prints an overview to the stdout.
  logging.info('\n'.join(contents))

  with tf.io.gfile.GFile(report_filename, 'w') as report_file:
    for dataset, supervision_mode in sorted(supervision_modes.items()):
      report_file.write('# Dataset: {} supervision_mode: {}\n'.format(
          dataset, supervision_mode))
    report_file.write('\n')
    report_file.write('\n'.join(contents))
    report_file.write('\n')

    for dataset, counter in counters.items():
      if not counter:
        continue
      report_file.write('# Detailed error statistics for {}:\n'.format(dataset))
      for key, value in sorted(counter.items()):
        report_file.write('# {}\t{}\n'.format(key, value))


def _parse_questions(interaction_dict,
                     supervision_modes,
                     report_filename):
  """Adds numeric value spans to all questions."""
  counters = collections.defaultdict(collections.Counter)
  # debug
  errs_map = {}
  for key, interactions in interaction_dict.items():
    # debug
    errs_map[key] = {}
    for interaction in interactions:
      questions = []
      for original_question in interaction.questions:
        try:
          question = interaction_utils_parser.parse_question(
              interaction.table, original_question, supervision_modes[key])
          counters[key]['valid'] += 1
        except ValueError as exc:
          question = interaction_pb2.Question()
          question.CopyFrom(original_question)
          question.answer.is_valid = False
          counters[key]['failed'] += 1
          counters[key]['failed-' + str(exc)] += 1
          # debug
          if str(exc) not in errs_map[key]:
            errs_map[key][str(exc)] = []
          errs_map[key][str(exc)].append(question.id)
                

        questions.append(question)

      del interaction.questions[:]
      interaction.questions.extend(questions)

  _write_report(report_filename, supervision_modes, counters)
  # debug
  with open(os.path.join(os.path.dirname(report_filename),'err_ids.json'),'w') as f:
    print('Writing invalid answer errors to file')
    json.dump(errs_map,f)
    time.sleep(5)
        


def _write_tfrecord(
    interactions,
    filepath,
    token_selector,
):
  with tf.io.TFRecordWriter(filepath + '.tfrecord') as writer:
    for interaction in interactions:
      if token_selector is not None:
        interaction = token_selector.annotated_interaction(interaction)
      writer.write(interaction.SerializeToString())


def _get_output_filename(output_dir, input_file):
  basename = os.path.splitext(input_file)[0]
  return os.path.join(output_dir, basename)


def create_interactions(
    supervision_modes,
    input_dir,
    output_dir,
    token_selector,
):
  """Converts data in SQA format to Interaction protos.

  Args:
    supervision_modes: Import for WikiSQL, decide if supervision is removed.
    input_dir: SQA data.
    output_dir: Where interactions will be written.
    token_selector: Optional helper class to keep more relevant tokens in input.
  """
  file_utils.make_directories(output_dir)

  interaction_dict = _read_interactions(input_dir)
  _add_tables(input_dir, interaction_dict)
  _parse_questions(interaction_dict, supervision_modes,
                   os.path.join(output_dir, 'report.tsv'))
# temp disable
  # print('temp disabled writing to tf records in sqa_utils.py:200')
  # time.sleep(3)
  for filename, interactions in interaction_dict.items():
    _write_tfrecord(interactions, _get_output_filename(output_dir, filename),
                    token_selector)

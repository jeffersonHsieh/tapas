
from typing import Iterable, List, Text

import hydra
from hydra.utils import get_original_cwd,to_absolute_path
from omegaconf import OmegaConf
import os


from tapas.protos import interaction_pb2
from tapas.retrieval import tfidf_baseline_utils
from tapas.retrieval.tfidf_baseline_utils import BM25Index, _tokenize, _remove_duplicates
from tapas.scripts import prediction_utils

import tqdm
import functools
import multiprocessing as mp
from multiprocessing import Pool
import json
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

config_dir = "/workspace/hsiehcc/tapas/hydraconfs"

def compute_jaccard(s1:set,s2:set):
    return len(s1 & s2) / len(s1 | s2)

def filter_table_acronym_by_query(
        question_token_set: set,
        table: interaction_pb2.Table, use_jaccard = False,
        jaccard_threshold: float = 0.5):
    if table.abbvs:
        for abbv in table.abbvs:
            abbv_toks = tfidf_baseline_utils._tokenize(abbv.abbreviation)
            if use_jaccard:
                if compute_jaccard(
                    set(abbv_toks),
                    question_token_set) >= jaccard_threshold:
                    yield abbv.abbreviation
            else:
                if set(abbv_toks) <= question_token_set:
                    yield abbv.abbreviation

def _iterate_weighted_table_texts(weight_cfg, table, question=None):
    for _ in range(weight_cfg.doctitle):
        if table.document_title:
            yield table.document_title
    for _ in range(weight_cfg.sectitle):
        if table.section_title:
            yield table.section_title
    for _ in range(weight_cfg.caption):
        if table.caption:
            yield table.caption
    for _ in range(weight_cfg.header):
        for column in table.columns:
            yield column.text
    for _ in range(weight_cfg.content):
        for row in table.rows:
            for cell in row.cells:
                yield cell.text
    # might be obsolete for v3
    if question:
        for _ in range(weight_cfg.abbv):
            yield from filter_table_acronym_by_query(
                question_token_set=set(_tokenize(question)),
                table=table,
                use_jaccard=weight_cfg.use_jaccard,
                jaccard_threshold=weight_cfg.abbv_filter_jaccard_threshold
            )
            
def _iterate_and_filter_table_abbv(weight_cfg,table,question):
    for _ in range(weight_cfg.abbv):
        yield from filter_table_acronym_by_query(
            question_token_set=set(_tokenize(question)),
            table=table,
            use_jaccard=weight_cfg.use_jaccard,
            jaccard_threshold=weight_cfg.abbv_filter_jaccard_threshold
        )

def add_table_abbv(weight_cfg, table, question, weighted_table_text):
    weighted_table_text_set = set(weighted_table_text)
    for text in _iterate_and_filter_table_abbv(weight_cfg,table,question):
        for tok in _tokenize(text):
            if tok not in weighted_table_text_set:
                weighted_table_text.append(tok)
                weighted_table_text_set.add(tok)


def create_uneven_bm25_index(
    cfg, tables, interaction=None
    ):
    """Creates a new index."""
    corpus = []
    table_ids = []
    question = interaction.questions[0].original_text if interaction else None
    for i,table in enumerate(tqdm.tqdm(_remove_duplicates(tables))):
        if cfg.index.bm25.expand_gold_only:
            if question is None:
                raise ValueError("question is None")
            if table.table_id == interaction.table.table_id:
                corpus.append(
            [tok for text in _iterate_weighted_table_texts(cfg.index.bm25,
                table, question) for tok in _tokenize(text)]
                )
            else:
                corpus.append(
            [tok for text in _iterate_weighted_table_texts(cfg.index.bm25,
                table, None) for tok in _tokenize(text)]
                )
        else:
            weighted_table_text = [tok for text in _iterate_weighted_table_texts(cfg.index.bm25,
                    table) for tok in _tokenize(text)]
            # ol = len(weighted_table_text)
            add_table_abbv(cfg.index.bm25, table, question, weighted_table_text)
            # if len(weighted_table_text)>ol:
            #     raise ValueError("weighted_table_text is longer than original")
            corpus.append(
                weighted_table_text
            )
        table_ids.append(table.table_id)

    return BM25Index(corpus, table_ids)

def evaluate_single(cfg, interaction, index):
    question = interaction.questions[0]
    question_text = question.original_text
    if cfg.retrieval.filter_stop_words:
        stw = set(stopwords.words('english'))
        question_tokens = [tok for tok in _tokenize(question.original_text) if tok not in stw]
        question_text = " ".join(question_tokens)
    scored_hits = index.retrieve(question_text)
    reference_table_id = interaction.table.table_id

    out = {}
    out['qid'] = interaction.id #question.id
    out['scores'] = scored_hits[:cfg.retrieval.max_table_rank]
    out['ref_table_id'] = reference_table_id

    gold_rank = None
    for rank, (table_id, _) in enumerate(scored_hits[:cfg.retrieval.max_table_rank]):
        if table_id == reference_table_id:
            gold_rank = rank
            break

    return gold_rank, out


def build_index_for_question(cfg, interaction, tables):
    if cfg.retrieval.use_bm25:
        return create_uneven_bm25_index(cfg, tables, 
            interaction=interaction)
    else:
        raise ValueError("Unknown retrieval method")

def build_and_eval_wrapper(cfg,interaction):
    if len(interaction.questions) != 1:
            raise ValueError(
                "Only one question per interaction supported"
            )
    
    index = build_index_for_question(
        cfg,
        interaction,
        tables=tfidf_baseline_utils.iterate_tables(to_absolute_path(cfg.data.tables_file))
    )

    gold_rank, scored_hits_and_metadata = evaluate_single(
            cfg, interaction, index
        )
    return gold_rank, scored_hits_and_metadata

@hydra.main(config_path= config_dir, config_name="table_acronym_oracle")
def main(cfg):
    # setup
    thresholds = [1, 5, 10, 15, cfg.retrieval.max_table_rank]
    interaction_file = cfg.data.interaction_file
    print(f"Test set: {interaction_file}")
    print(OmegaConf.to_yaml(cfg))
    interactions = list(prediction_utils.iterate_interactions(to_absolute_path(interaction_file)))
    

    ranks = []
    scored_hits_list = []
    # iterate through questions
    build_and_eval_func = functools.partial(build_and_eval_wrapper, cfg)
    if cfg.parallel:
        with Pool(mp.cpu_count()) as P:
            ranks_and_scored_hits = list(tqdm.tqdm(P.imap(build_and_eval_func, interactions),total=len(interactions)))
        ranks, scored_hits_list = list(map(list, zip(*ranks_and_scored_hits)))
        ranks = list(filter(lambda x:x is not None, ranks))
    else:
        for interaction in tqdm.tqdm(interactions):
            gold_rank, scored_hits_and_metadata = build_and_eval_func(interaction)
            scored_hits_list.append(scored_hits_and_metadata)
            if gold_rank is not None:
                ranks.append(gold_rank)
        

    def precision_at_th(threshold):
        return sum(1 for rank in ranks if rank < threshold) / len(interactions)
    values = [f"{precision_at_th(threshold):.4}" for threshold in thresholds]


    
    with open(os.path.join(os.getcwd(), "results.json"), "w") as f:
        json.dump(scored_hits_list, f)
    with open(os.path.join(os.getcwd(), "scores.txt"), "w") as f:
        f.write(",".join(list(map(str,thresholds))) + "\n")
        f.write(",".join(values))
    print(OmegaConf.to_yaml(cfg))
    print("\t".join(list(map(str,thresholds)))+"\n"+"\t".join(values))
    print("Working directory : {}".format(os.getcwd()))

if __name__ == "__main__":
    main()
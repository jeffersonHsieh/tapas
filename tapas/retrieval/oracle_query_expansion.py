
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



def filter_table_acronym_by_query(
        question_token_set: set,
        table: interaction_pb2.Table, skip_existed=True):
    if table.abbvs:
        for abbv in table.abbvs:
            abbv_toks = tfidf_baseline_utils._tokenize(abbv.abbreviation)
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
    if question is not None:
        for _ in range(weight_cfg.abbv):
            yield from filter_table_acronym_by_query(
                question_token_set=set(_tokenize(question)),
                table=table,
                use_jaccard=weight_cfg.use_jaccard,
                jaccard_threshold=weight_cfg.abbv_filter_jaccard_threshold
            )
            


def create_uneven_bm25_index(
    cfg, tables, question=None
):
    """Creates a new index."""
    corpus = []
    table_ids = []

    for i,table in enumerate(tqdm.tqdm(_remove_duplicates(tables))):
        corpus.append(
            [tok for text in _iterate_weighted_table_texts(cfg.index.bm25,
                table, question) for tok in _tokenize(text)]
        )
        table_ids.append(table.table_id)

    return BM25Index(corpus, table_ids)

def expand_question(cfg,interaction, question):
    question_tokens = _tokenize(question.original_text)
    ref_table_toks = [tok for text in _iterate_weighted_table_texts(cfg.index.bm25,
                interaction.table, question) for tok in _tokenize(text)]
    if cfg.retrieval.filter_stop_words:
        stw = set(stopwords.words('english'))
        # if cfg.retrieval.filter_stop_words_v1:
        #     question_tokens = [tok for tok in question_tokens if tok not in stw]
        # else:
        question_tokens = [tok for tok in question_tokens if tok not in (stw-set(ref_table_toks))]

    augmented_question_toks = []
    abbv_map = {abbv.abbreviation.lower(): abbv.expansion for abbv in interaction.table.abbvs}
    if cfg.retrieval.expand_abbv:
        for tok in question_tokens:
            augmented_question_toks.append(tok)
            if abbv_map and cfg.debug:
                import pdb;pdb.set_trace()
            if tok in abbv_map and tok not in set(ref_table_toks):
                augmented_question_toks.extend(_tokenize(abbv_map[tok]))
    else:
        augmented_question_toks = question_tokens
    if cfg.retrieval.expand_ref_doc_title:
        augmented_question_toks.extend(_tokenize(interaction.table.document_title))
    if cfg.retrieval.expand_ref_sec_title:
        augmented_question_toks.extend(_tokenize(interaction.table.section_title))
    if cfg.retrieval.expand_ref_caption:
        augmented_question_toks.extend(_tokenize(interaction.table.caption))
    return ' '.join(augmented_question_toks)

def peek_and_evaluate_single(cfg, index, interaction):
    question = interaction.questions[0]
    expanded_question_text = expand_question(cfg, interaction, question)
    scored_hits = index.retrieve(expanded_question_text)
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


def build_index(cfg, tables):
    if cfg.retrieval.use_bm25:
        return create_uneven_bm25_index(cfg, tables)
    else:
        raise ValueError("Unknown retrieval method")



@hydra.main(config_path= config_dir, config_name="query_expansion_oracle")
def main(cfg):
    # setup
    thresholds = [1, 5, 10, 15, cfg.retrieval.max_table_rank]
    interaction_file = cfg.data.interaction_file
    print(f"Test set: {interaction_file}")
    interactions = list(prediction_utils.iterate_interactions(to_absolute_path(interaction_file)))
    

    ranks = []
    scored_hits_list = []

    # build index
    index = build_index(
        cfg,
        tables=tfidf_baseline_utils.iterate_tables(to_absolute_path(cfg.data.tables_file))
    )    

    # expand and evaluate
    eval_func = functools.partial(peek_and_evaluate_single, cfg, index)
    if cfg.parallel:
        with Pool(mp.cpu_count()) as P:
            ranks_and_scored_hits = list(tqdm.tqdm(P.imap(eval_func, interactions),total=len(interactions)))
        ranks, scored_hits_list = list(map(list, zip(*ranks_and_scored_hits)))
        ranks = list(filter(lambda x:x is not None, ranks))
    else:
        for interaction in tqdm.tqdm(interactions):
            gold_rank, scored_hits_and_metadata = eval_func(interaction)
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
import os
import sys
from argparse import ArgumentParser
from typing import List

import torch

from deepkin.clib.libkinlp.kinlpy import ParsedMorphoToken, BOS_ID, build_kinlpy_lib, \
    parse_text_to_morpho_sentence
from deepkin.models.arguments import py_trainer_args
from deepkin.models.kinyabert import KinyaBERT_SequenceTagger
from deepkin.models.modules import BaseConfig


class NERTaggingEngine:
    def __init__(self, args, device, tagger_model, tag_label_dict, lib, ffi):
        self.args = args
        self.device = device
        self.ner_tagger_model = tagger_model
        self.tag_label_dict = tag_label_dict
        self.tag_dict = {k:v for k, v in enumerate(tag_label_dict)}
        self.lib = lib
        self.ffi = ffi

def ner_tagging_engine_setup(rank=0) -> NERTaggingEngine:
    args = py_trainer_args(list_args=[], silent=True)
    if args.gpus == 0:
        args.world_size = 1
    cfg = BaseConfig()

    USE_GPU = (args.gpus > 0)
    device = torch.device('cpu')
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(rank))
    tag_label_dict = {}
    tag_label_dict['B-PER'] = 0
    tag_label_dict['I-PER'] = 1
    tag_label_dict['B-ORG'] = 2
    tag_label_dict['I-ORG'] = 3
    tag_label_dict['B-LOC'] = 4
    tag_label_dict['I-LOC'] = 5
    tag_label_dict['B-DATE'] = 6
    tag_label_dict['I-DATE'] = 7
    tag_label_dict['O'] = 8
    KINLP_HOME = (os.environ['KINLP_HOME']) if ('KINLP_HOME' in os.environ) else ('/opt/KINLP')
    tagger_model = KinyaBERT_SequenceTagger(args, cfg, len(tag_label_dict)).to(device)
    kb_state_dict = torch.load(KINLP_HOME+'/models/NER_kinyabert_base_2023-10-16.pt', map_location=device)
    tagger_model.load_state_dict(kb_state_dict['model_state_dict'])
    del kb_state_dict
    tagger_model.eval()

    build_kinlpy_lib()
    from kinlpy import ffi, lib

    lib.init_kinlp_socket()

    return NERTaggingEngine(args, device, tagger_model, tag_label_dict, lib, ffi)

def prepare_ner_input_segments(tokens: List[ParsedMorphoToken], max_len=512):
    lm_morphs = []
    pos_tags = []
    affixes = []
    tokens_lengths = []
    stems = []
    extended = []

    # Add <CLS> Token
    lm_morphs.append(BOS_ID)
    pos_tags.append(BOS_ID)
    stems.append(BOS_ID)
    tokens_lengths.append(0)

    for token in tokens:
        if (len(tokens_lengths) >= max_len):
            break
        lm_morphs.append(token.lm_morph_id)
        pos_tags.append(token.pos_tag_id)
        stems.append(token.stem_id)
        extended.append(False)
        affixes.extend(token.affixes)
        tokens_lengths.append(len(token.affixes))
        for tid in token.extra_tokens_ids:
            if (len(tokens_lengths) >= max_len):
                break
            lm_morphs.append(token.lm_morph_id)
            pos_tags.append(token.pos_tag_id)
            stems.append(tid)
            extended.append(True)
            affixes.extend(token.affixes)
            tokens_lengths.append(len(token.affixes))

    assert len(affixes) == sum(tokens_lengths), "@prepare_ner_input_segments: Mismatch token lengths affixes={} vs lengths={}".format(
        len(affixes), sum(tokens_lengths))
    return (lm_morphs,
            pos_tags,
            affixes,
            tokens_lengths,
            stems,
            extended)

def ner_data_collate_wrapper(data_item):
    (seq_lm_morphs,
     seq_pos_tags,
     seq_affixes,
     seq_tokens_lengths,
     seq_stems,
     seq_extended) = data_item

    return (seq_lm_morphs,
            seq_pos_tags,
            seq_affixes,
            seq_tokens_lengths,
            seq_stems,
            seq_extended,
            [len(seq_tokens_lengths)])

def ner_tag_model_batch_predict(ner_batch_data_item, ner_model: KinyaBERT_SequenceTagger, device):
    (batch_lm_morphs,
     batch_pos_tags,
     batch_affixes,
     batch_tokens_lengths,
     batch_stems,
     batch_extended,
     batch_input_sequence_lengths) = ner_batch_data_item

    tokens_lengths = batch_tokens_lengths
    input_sequence_lengths = batch_input_sequence_lengths

    lm_morphs = torch.tensor(batch_lm_morphs).to(device)
    pos_tags = torch.tensor(batch_pos_tags).to(device)
    affixes = torch.tensor(batch_affixes).to(device)
    stems = torch.tensor(batch_stems).to(device)

    scores = ner_model.forward(lm_morphs, pos_tags, affixes, tokens_lengths, stems, input_sequence_lengths)
    predicted_labels_idx = torch.argmax(scores, dim=1)
    seq_len = [(leng-1) for leng in input_sequence_lengths]
    labels = [lab.cpu().tolist() for lab in predicted_labels_idx.split(seq_len)]
    return labels

def ner_tagging(ner_tagging_engine: NERTaggingEngine, txt: str):
    parsed_sentence = parse_text_to_morpho_sentence(ner_tagging_engine.ffi, ner_tagging_engine.lib, txt)
    batch_data_item = ner_data_collate_wrapper(prepare_ner_input_segments(parsed_sentence.tokens))
    batch_extended = batch_data_item[-2]
    ner_labels = ner_tag_model_batch_predict(batch_data_item, ner_tagging_engine.ner_tagger_model, ner_tagging_engine.device)
    ner_tags = [ner_tagging_engine.tag_dict[tid] for tid in ner_labels[0]]
    assert len(ner_tags) == len(batch_extended), f"TAG/EXTENDED labels lengths mismatch!: {len(ner_tags)}: {ner_tags} vs {len(batch_extended)}: {batch_extended}"
    ner_tags = [tag for i,tag in enumerate(ner_tags) if not batch_extended[i]]
    raw_tokens = [token.raw_surface_form for token in parsed_sentence.tokens]
    assert len(ner_tags)==len(raw_tokens), f"TAG/TOKEN lengths mismatch!: {len(ner_tags)}: {ner_tags} vs {len(raw_tokens)}: {raw_tokens}"
    return [(tag,token) for tag,token in zip(ner_tags,raw_tokens)]

def ner_inference_main(input_text:str):
    import time
    ner_tagging_engine = ner_tagging_engine_setup()

    t0 = time.perf_counter()
    tags_and_tokens = ner_tagging(ner_tagging_engine, input_text)
    t1 = time.perf_counter()
    while True:
        print(f'\n\nInput: {input_text}\n\n')
        for tag,token in tags_and_tokens:
            if (tag != 'O'):
                if tag[:1] == 'B':
                    print('')
                print(f'{tag}: {token}')
        print('\n\nNER Tagging took: {:.0f} ms\n'.format(1000.0 * (t1 - t0)))
        input_text = input("\nInput more text(e/E/q/Q to Exit): ")
        if input_text in {'exit', 'EXIT', 'e', 'E', 'quit', 'QUIT', 'q', 'Q'}:
            print('Exiting ...')
            sys.exit(0)
        t0 = time.perf_counter()
        tags_and_tokens = ner_tagging(ner_tagging_engine, input_text)
        t1 = time.perf_counter()

if __name__ == '__main__':
    parser = ArgumentParser(description="NER Inference arguments")
    parser.add_argument("--input", type=str, default=None)
    local_arguments = parser.parse_args()
    ner_inference_main(local_arguments.input)

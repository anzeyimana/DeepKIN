from typing import List

import progressbar
from torch.utils.data import Dataset

from deepkin.models.data import prepare_mlm_data_from_sentence
from deepkin.models.modules import BaseConfig
from deepkin.utils.misc_functions import read_lines, normalize_kinya_text
from deepkin.clib.libkinlp.kinlpy import ParsedMorphoSentence, BOS_ID, EOS_ID, parse_text_to_morpho_sentence

def prepare_input_segments(input_segments: List[ParsedMorphoSentence], max_len=512):
    lm_morphs = []
    pos_tags = []
    affixes = []
    tokens_lengths = []
    stems = []

    # Add <CLS> Token
    lm_morphs.append(BOS_ID)
    pos_tags.append(BOS_ID)
    stems.append(BOS_ID)
    tokens_lengths.append(0)

    started = False
    for segment in input_segments:
        if started:
            # Add <SEP> Token
            lm_morphs.append(EOS_ID)
            pos_tags.append(EOS_ID)
            stems.append(EOS_ID)
            tokens_lengths.append(0)
        for token in segment.tokens:
            if (len(tokens_lengths) >= max_len):
                break
            lm_morphs.append(token.lm_morph_id)
            pos_tags.append(token.pos_tag_id)
            stems.append(token.stem_id)
            affixes.extend(token.affixes)
            tokens_lengths.append(len(token.affixes))
            for tid in token.extra_tokens_ids:
                if (len(tokens_lengths) >= max_len):
                    break
                lm_morphs.append(token.lm_morph_id)
                pos_tags.append(token.pos_tag_id)
                stems.append(tid)
                affixes.extend(token.affixes)
                tokens_lengths.append(len(token.affixes))
        if (len(tokens_lengths) >= max_len):
            break
        started = True

    assert len(affixes) == sum(tokens_lengths), "@prepare_cls_reg_input_segments: Mismatch token lengths affixes={} vs lengths={}".format(
        len(affixes), sum(tokens_lengths))
    return (lm_morphs,
            pos_tags,
            affixes,
            tokens_lengths,
            stems)
def parse_corpus(corpus_file) -> List[ParsedMorphoSentence]:
    # build_kinlpy_lib()
    from kinlpy import ffi, lib
    lib.init_kinlp_socket()
    ret = []
    print(f'Parsing corpus file: {corpus_file} ...')
    lines = read_lines(corpus_file)
    print(f'Got {len(lines)} lines!')
    with progressbar.ProgressBar(max_value=len(lines), redirect_stdout=True) as bar:
        for itr,line in enumerate(lines):
            if (itr % 100) == 0:
                bar.update(itr)
            line = line.strip('\n')
            line = normalize_kinya_text(line)
            if len(line) > 5:
                if len(line.split()) > 4:
                    sent = parse_text_to_morpho_sentence(ffi, lib, line)
                    length = sum([len(t.extra_tokens_ids)+1 for t in sent.tokens])
                    if (length > 4) and (length < 510):
                        ret.append(sent)
    print(f'Got {len(ret)} parsed sentences!')

    return ret

class MyMLMDataset(Dataset):

    def __init__(self, corpus_file):
        self.cfg = BaseConfig()
        self.parsed_sentences = parse_corpus(corpus_file)

    def __len__(self):
        return len(self.parsed_sentences)

    def __getitem__(self, idx):
        while True:
            sample = prepare_mlm_data_from_sentence(self.parsed_sentences[idx], True, self.cfg)
            if len(sample[7])>0:
                return (0,sample)
            idx = (idx + 1) % len(self.parsed_sentences)

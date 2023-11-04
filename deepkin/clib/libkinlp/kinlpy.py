import os
from typing import List
import distro

import youtokentome as yttm
from cffi import FFI

PAD_ID = 0
UNK_ID = 1
MSK_ID = 2
BOS_ID = 3
EOS_ID = 4

NUM_SPECIAL_TOKENS = 5
MY_PRINTABLE = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

def build_kinlpy_lib():
    ffibuilder = FFI()

    KINLP_HOME = (os.environ['KINLP_HOME']) if ('KINLP_HOME' in os.environ) else ('/opt/KINLP')

    ffibuilder.cdef("""
        void init_kinlp_socket(void);
        char * synth_morpho_token_via_socket(const char * wt_idx, const char * stem, const char * fsa_key, const char * indices_csv);
        char * kinlpy_parse_text_via_socket(const char * text);
        void free_token(char * token);
    """)

    ffibuilder.set_source("kinlpy",
                          """
                               #include \""""+KINLP_HOME+"""/include/kinlpy.h"
                          """,
                          extra_compile_args=['-fopenmp', '-D use_openmp', '-O3', '-march=x86-64', '-ffast-math',
                                              '-Wall', '-Werror', "-Wl,-rpath,'" + KINLP_HOME + "/lib'"],
                          extra_link_args=['-fopenmp'],
                          library_dirs=[(KINLP_HOME + '/lib/'+distro.version())],
                          libraries=['morphokin'])  # library name, for the linker

    ffibuilder.compile(verbose=False)

class ParsedToken:
    def __init__(self, w, ffi):
        # POS Info
        self.lm_stem_id = w.lm_stem_id
        self.lm_morph_id = w.lm_morph_id
        self.pos_tag_id = w.pos_tag_id
        self.valid_orthography = w.surface_form_has_valid_orthography

        # Morphology
        self.stem_id = w.stem_id
        self.affix_ids = [w.affix_ids[i] for i in range(w.len_affix_ids)]
        self.extra_stem_token_ids = [w.extra_stem_token_ids[i] for i in range(w.len_extra_stem_token_ids)]

        # Text
        self.is_apostrophed = w.is_apostrophed
        self.surface_form = ffi.string(w.surface_form).decode("utf-8") if (w.len_surface_form > 0) else ''
        self.raw_surface_form = ffi.string(w.raw_surface_form).decode("utf-8") if (w.len_raw_surface_form > 0) else ''
        if (self.is_apostrophed != 0) and (len(self.raw_surface_form) > 0) and ((self.raw_surface_form[-1] == 'a') or (self.raw_surface_form[-1] == 'A')):
            self.raw_surface_form = self.raw_surface_form[:-1]+"\'"

    def to_parsed_format(self) -> str:
        word_list = [self.lm_stem_id, self.lm_morph_id, self.pos_tag_id, self.stem_id] + [len(self.extra_stem_token_ids)] + self.extra_stem_token_ids + [len(self.affix_ids)] + self.affix_ids
        return ','.join([str(x) for x in word_list])

DEL_SYMBOL = '392THEWIUT43'
PIP_SYMBOL = 'SDOI2HFKJEE'
class ParsedMorphoToken:
    def __init__(self, parsed_token:str, real_parsed_token : ParsedToken = None, delimiter=';'):
        if real_parsed_token is not None:
            self.lm_stem_id = real_parsed_token.lm_stem_id
            self.lm_morph_id = real_parsed_token.lm_morph_id
            self.pos_tag_id = real_parsed_token.pos_tag_id
            self.stem_id = real_parsed_token.stem_id # My God: This bug here almost caused me panic! lm_stem_id should be different from stem_id
            self.extra_tokens_ids = real_parsed_token.extra_stem_token_ids
            self.affixes = real_parsed_token.affix_ids
            self.is_apostrophed = real_parsed_token.is_apostrophed
            self.surface_form = real_parsed_token.surface_form
            self.raw_surface_form = real_parsed_token.raw_surface_form
        else:
            parsed_token = parsed_token.replace('||','|'+PIP_SYMBOL)
            self.surface_form = '_'
            self.raw_surface_form = '_'
            if '|' in parsed_token:
                idx = parsed_token.index('|')
                tks = parsed_token[:idx].split(',')
                sfc = parsed_token[(idx + 1):]
                if sfc == DEL_SYMBOL:
                    sfc = delimiter
                if sfc == PIP_SYMBOL:
                    sfc = '|'
                if len(sfc) > 0:
                    self.surface_form = sfc.lower()
                    self.raw_surface_form = sfc
            else:
                tks = parsed_token.split(',')
            self.lm_stem_id = int(tks[0])
            self.lm_morph_id = int(tks[1])
            self.pos_tag_id = int(tks[2])
            self.stem_id = int(tks[3])
            num_ext = int(tks[4])
            self.extra_tokens_ids = [int(v) for v in tks[5:(5+num_ext)]]
            # This is to cap too long tokens for position encoding
            self.extra_tokens_ids = self.extra_tokens_ids[:64]
            num_afx = int(tks[(5+num_ext)])
            self.affixes = [int(v) for v in tks[(6+num_ext):(6+num_ext+num_afx)]]
            self.is_apostrophed = 0

    def to_parsed_format(self) -> str:
        word_list = [self.lm_stem_id, self.lm_morph_id, self.pos_tag_id, self.stem_id] + [len(self.extra_tokens_ids)] + self.extra_tokens_ids + [len(self.affixes)] + self.affixes
        return (','.join([str(x) for x in word_list])) + '|' + self.raw_surface_form
    def get_surface_forms(self, bpe:yttm.BPE):
        if len(self.extra_tokens_ids) > 0:
            tkns = bpe.encode([self.raw_surface_form],
                              output_type=yttm.OutputType.SUBWORD, bos=False, eos=False, reverse=False,
                             dropout_prob=0)[0]
            return [(k if (k[0] == '▁') else ('@@' + k)) for k in tkns]
        else:
            return [self.raw_surface_form]


class ParsedMorphoSentence:
    def __init__(self, parsed_sentence_line:str, parsed_tokens: List[ParsedToken] = None, delimiter=';'):
        if parsed_tokens is not None:
            self.tokens = [ParsedMorphoToken('_', real_parsed_token=token) for token in parsed_tokens]
        else:
            parsed_sentence_line = parsed_sentence_line.replace(delimiter + delimiter, DEL_SYMBOL + delimiter)
            if parsed_sentence_line.endswith(delimiter):
                parsed_sentence_line = parsed_sentence_line[:-1] + DEL_SYMBOL
            self.tokens = [ParsedMorphoToken(v, delimiter=delimiter) for v in parsed_sentence_line.split(delimiter) if len(v)>0]
    def to_parsed_format(self) -> str:
        return ';'.join([tk.to_parsed_format() for tk in self.tokens])

def parse_text_to_morpho_sentence(ffi, lib, txt: str) -> ParsedMorphoSentence:
    if not any(c in MY_PRINTABLE for c in txt):
        return ParsedMorphoSentence(None, parsed_tokens = [], delimiter=';')
    ret_str = lib.kinlpy_parse_text_via_socket(txt.encode('utf-8'))
    parsed_sentence_line = ffi.string(ret_str).decode("utf-8")
    lib.free_token(ret_str)
    return ParsedMorphoSentence(parsed_sentence_line)

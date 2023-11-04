from typing import Union

import numpy as np
import math
from deepkin.clib.libkinlp.kinlpy import NUM_SPECIAL_TOKENS
from deepkin.clib.libkinlp.language_data import all_wt_abbrevs, all_word_types, all_pos_tags, all_wt_abbrevs_dict

special_tokens = ['<pad>', '<unk>', '<msk>', '<s>', '</s>']
MY_PRINTABLE = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

class Affix():
    def __init__(self, id, line):
        tk = line.split(':')
        self.id = id
        self.wt = int(tk[0])
        self.slot = int(tk[1])
        self.idx = int(tk[2])
        self.key = ':'.join(tk[:3])
        self.prob = float(tk[3])
        self.view = all_wt_abbrevs[self.wt] + ':' + str(self.slot) + ':' + \
                    all_word_types[self.wt]["morpheme_sets"][self.slot][self.idx]


class Afset():
    def __init__(self, id, line):
        tk = line.split(':')
        self.id = id
        self.wt = int(tk[0])
        self.fsa = int(tk[1])
        self.len_indices = int(tk[2])
        self.indices = [int(s) for s in tk[3].split(',')]
        self.prob = float(tk[4])
        self.slots = all_word_types[self.wt]["morphotactics_fsa"][self.fsa]
        self.affixes_keys = [(str(self.wt) + ':' + str(s) + ':' + str(i)) for s, i in zip(self.slots, self.indices) if (
                    (all_word_types[self.wt]["stem_start_idx"] != s) and (
                        all_word_types[self.wt]["stem_end_idx"] != s))]
        self.view = all_wt_abbrevs[self.wt] + ':' + '-'.join([(all_word_types[self.wt]["morpheme_sets"][s][i]) if (
                    (all_word_types[self.wt]["stem_start_idx"] != s) and (
                        all_word_types[self.wt]["stem_end_idx"] != s)) else "*" for s, i in
                                                              zip(self.slots, self.indices)])


def read_all_affixes(fn):
    with open(fn) as f:
        affixes = [Affix(i, line.rstrip('\n')) for i, line in enumerate(f) if len(line.rstrip('\n')) > 0]
    return affixes


def read_corr_table(fn):
    ret = dict()
    with open(fn) as f:
        for line in f:
            ln = line.rstrip('\n')
            if len(ln) > 0:
                t = ln.split(',')
                ret[t[0]] = float(t[1])
    return ret


def read_all_afsets(fn):
    with open(fn) as f:
        afsets = [Afset(i, line.rstrip('\n')) for i, line in enumerate(f) if len(line.rstrip('\n')) > 0]
    return afsets


def affix_view(id, all_affixes):
    if id < 5:
        return special_tokens[id]
    else:
        return all_affixes[id - 5].view


def id_to_affix(id, all_affixes) -> Union[Affix,None]:
    if id < 5:
        return None
    else:
        return all_affixes[id - 5]

morph_pos_tags = 0

def afset_view(id, all_afsets):
    global morph_pos_tags
    if id < 5:
        return special_tokens[id]
    elif (id - 5) < len(all_afsets):
        return all_afsets[id - 5].view
    else:
        if morph_pos_tags == 0:
            morph_pos_tags = len([p for p in all_pos_tags if p["type"] == 0])
        if (id - len(all_afsets) + morph_pos_tags - 5) < len(all_pos_tags):
            return pos_tag_view(id - len(all_afsets) + morph_pos_tags)
        else:
            return f'[*unk*:{id}]'


def id_to_afset(id, all_afsets) -> Union[Afset,None]:
    global morph_pos_tags
    if id < 5:
        return None
    elif (id - 5) < len(all_afsets):
        return all_afsets[id - 5]
    else:
        return None


def pos_tag_view(id):
    if id < 5:
        return special_tokens[id]
    elif (id - 5) < len(all_pos_tags):
        return all_pos_tags[id - 5]["name"] + '#' + '{:03d}'.format(all_pos_tags[id - 5]["idx"])
    else:
        return f'[*unk*:{id}]'

def pos_tag_initials(id):
    if id < 5:
        return special_tokens[id]
    elif (id - 5) < len(all_pos_tags):
        return all_pos_tags[id - 5]["name"]
    else:
        return '<rare>'

def make_surface_form(stem_id, affix_ids, stems_vocab, all_affixes, ffi, lib):
    if len(affix_ids) > 0:
        affixes_list = [id_to_affix(id, all_affixes) for id in affix_ids]
        wt_idx = affixes_list[0].wt
        stem_slot = all_word_types[wt_idx]['stem_start_idx']
        slots_idx = [(x.slot, x.idx) for x in affixes_list]
        slots = set([x.slot for x in affixes_list])
        if stem_slot not in slots:
            slots_idx = slots_idx + [(stem_slot, 0)]
        # Handle verb reduplication
        if (wt_idx == 0) and (12 in slots) and (11 not in slots):
            slots_idx = slots_idx + [(11, 0)]
        slots_idx = sorted(slots_idx, key=lambda x: x[0], reverse=False)
        stem = stems_vocab[stem_id]
        if stem.find(':') > 0:
            stem = stem.split(':')[1]
        fsa_key = str(wt_idx) + ':' + '-'.join([str(x[0]) for x in slots_idx])
        indices_csv = ','.join([str(x[1]) for x in slots_idx])
        ret_str = lib.synth_morpho_token_via_socket(f'{wt_idx}'.encode('utf-8'), stem.encode('utf-8'), fsa_key.encode('utf-8'), indices_csv.encode('utf-8'))
        if ret_str == ffi.NULL:
            ret_str = '-'.join([affix_view(af.id + 5, all_affixes) for af in
                                affixes_list]) + '/' + stem + "({},'{}','{}','{}')".format(wt_idx, stem, fsa_key,
                                                                                           indices_csv)
            return ret_str, False
        surface_form = ffi.string(ret_str).decode("utf-8")
        lib.free_token(ret_str)
        if not any(c in MY_PRINTABLE for c in surface_form):
            ret_str = '-'.join([affix_view(af.id + 5, all_affixes) for af in
                                affixes_list]) + '/' + stem + "({},'{}','{}','{}')".format(wt_idx, stem, fsa_key,
                                                                                           indices_csv)
            return ret_str, False
        else:
            return (surface_form), True
    else:
        surface_form = stems_vocab[stem_id]
        if surface_form is not None:
            if (':' in surface_form) and (len(surface_form) > 2):
                cdx = surface_form.index(':')
                if cdx < (len(surface_form) - 1):
                    surface_form = surface_form[(cdx + 1):]
        return surface_form, True


def decode_word_per_wt(top_stems, top_pos_tags, top_afsets, top_affixes,
                       top_stems_prob, top_pos_tags_prob, top_afsets_prob, top_affixes_prob,
                       stems_vocab, all_afsets, all_affixes, all_afsets_inverted_index,
                       pos_afset_corr, pos_stem_corr, afset_stem_corr, afset_affix_slot_corr,
                       ffi, lib, wt, wt_prob,
                       prob_cutoff=0.3, affix_prob_cutoff=0.3, affix_min_prob=0.3):
    # 2. wt filtering
    stems = []
    pos_tags = []
    afsets = []
    affixes = []

    stems_prob = []
    pos_tags_prob = []
    afsets_prob = []
    affixes_prob = []

    for id, p in zip(top_stems, top_stems_prob):
        if stems_vocab[id].find(':') > 0:
            t = all_wt_abbrevs_dict[stems_vocab[id].split(':')[0]]
            if (wt == t):
                stems.append(id)
                stems_prob.append(p)
        elif (wt == -1):
            stems.append(id)
            stems_prob.append(p)

    for id, p in zip(top_pos_tags, top_pos_tags_prob):
        if (id >= 5) and ((id - 5) < len(all_pos_tags)):
            nm = all_pos_tags[id - 5]["name"]
            if nm in all_wt_abbrevs_dict:
                t = all_wt_abbrevs_dict[nm]
                if (wt == t):
                    pos_tags.append(id)
                    pos_tags_prob.append(p)
            elif (wt == -1):
                pos_tags.append(id)
                pos_tags_prob.append(p)
        elif (wt == -1):
            pos_tags.append(id)
            pos_tags_prob.append(p)

    for id, p in zip(top_afsets, top_afsets_prob):
        af = id_to_afset(id, all_afsets)
        if af is not None:
            if (wt == af.wt):
                afsets.append(id)
                afsets_prob.append(p)
        elif (wt == -1):
            afsets.append(id)
            afsets_prob.append(p)

    for id, p in zip(top_affixes, top_affixes_prob):
        af = id_to_affix(id, all_affixes)
        if af is not None:
            if (wt == af.wt):
                affixes.append(id)
                affixes_prob.append(p)

    # if (len(stems)==0):
    #     print('~~~~~~~~~~~~~~~> Stems wt filter failed')
    # if (len(pos_tags)==0):
    #     print('~~~~~~~~~~~~~~~> PosTags wt filter failed')
    # if (len(afsets)==0):
    #     print('~~~~~~~~~~~~~~~> Afsets wt filter failed')

    # 3. prob cut-off
    stems_cut = []
    pos_tags_cut = []
    afsets_cut = []
    affixes_cut = []

    stems_prob_cut = []
    pos_tags_prob_cut = []
    afsets_prob_cut = []
    affixes_prob_cut = []

    i = 0
    pp = 0.0
    for id, p in zip(stems, stems_prob):
        if (i > 0) and ((pp - p) >= prob_cutoff):
            break
        stems_cut.append(id)
        stems_prob_cut.append(p)
        #         print(f'STEM: {id} :> {p} :> {math.log(p)}')
        i += 1
        pp = p

    i = 0
    pp = 0.0
    for id, p in zip(pos_tags, pos_tags_prob):
        if (i > 0) and ((pp - p) >= prob_cutoff):
            break
        pos_tags_cut.append(id)
        pos_tags_prob_cut.append(p)
        #         print(f'POS: {id} :> {p} :> {math.log(p)}')
        i += 1
        pp = p

    i = 0
    pp = 0.0
    for id, p in zip(afsets, afsets_prob):
        if (i > 0) and ((pp - p) >= prob_cutoff):
            break
        afsets_cut.append(id)
        afsets_prob_cut.append(p)
        #         print(f'AFS: {id} :> {p} :> {math.log(p)}')
        i += 1
        pp = p

    i = 0
    pp = 0.0
    for id, p in zip(affixes, affixes_prob):
        if ((i > 0) and ((pp - p) >= affix_prob_cutoff)) or (p < affix_min_prob):
            break
        affixes_cut.append(id)
        affixes_prob_cut.append(p)
        i += 1
        pp = p

    # if (len(stems_cut)==0):
    #     print('~~~~~~~~~~~~~~~> Stems cut failed')
    # if (len(pos_tags_cut)==0):
    #     print('~~~~~~~~~~~~~~~> PosTags cut failed')
    # if (len(afsets_cut)==0):
    #     print('~~~~~~~~~~~~~~~> Afsets cut failed')

    # 5. fsa filtering
    afset_own_affixes = dict()
    for afs_id in afsets_cut:
        non_afset_affixes_stats = 0.0
        afset_affixes = []
        afset_affixes_slots = set()
        afs = id_to_afset(afs_id, all_afsets)
        if afs is not None:
            for k in afs.affixes_keys:
                afx = all_afsets_inverted_index[k]
                if afx.slot not in afset_affixes_slots:
                    afset_affixes.append(afx.id + NUM_SPECIAL_TOKENS)
                    afset_affixes_slots.add(afx.slot)
        # affix slot conflict resolution
        for a, p in zip(affixes_cut, affixes_prob_cut):
            afx = id_to_affix(a, all_affixes)
            if afx is not None:
                af_key = '{}-{}:{}'.format(afs_id, afx.wt, afx.slot)
                if (afx.slot not in afset_affixes_slots) and (af_key in afset_affix_slot_corr):
                    afset_affixes.append(a)
                    afset_affixes_slots.add(afx.slot)
                    non_afset_affixes_stats += 1.0
        afset_own_affixes[afs_id] = afset_affixes, non_afset_affixes_stats

    # 5. Apply correlations
    corr = np.zeros((len(stems_cut), len(pos_tags_cut), len(afsets_cut)))  # + 1e-7

    for si, s in enumerate(stems_cut):
        #         print(f'STEM>> {s} --> {stems_prob_cut[si]}')
        for pi, p in enumerate(pos_tags_cut):
            for ai, a in enumerate(afsets_cut):
                ps_key = '{}-{}'.format(p, s)
                pa_key = '{}-{}'.format(p, a)
                as_key = '{}-{}'.format(a, s)
                if (ps_key in pos_stem_corr) and (pa_key in pos_afset_corr) and (as_key in afset_stem_corr):
                    #                     corr[si,pi,ai] += math.exp((pos_stem_corr[ps_key]*0.1)+
                    #                                                (pos_afset_corr[pa_key]*0.1)+
                    #                                                (afset_stem_corr[as_key]*0.1)+
                    #                                                math.log(stems_prob_cut[si])+
                    #                                                math.log(pos_tags_prob_cut[pi])+
                    #                                                math.log(afsets_prob_cut[ai]))
                    #                     if (pos_stem_corr[ps_key] > -100.0) and (pos_afset_corr[pa_key] > -10.0) and (afset_stem_corr[as_key] > -10.0):
                    corr[si, pi, ai] = max(corr[si, pi, ai],
                                           math.exp((afset_stem_corr[as_key]*0.08) +
                                                    math.log(stems_prob_cut[si]) +
                                                    math.log(pos_tags_prob_cut[pi]) +
                                                    math.log(afsets_prob_cut[ai])))

    results = {}
    corr_Z = corr.sum()
    for si, s in enumerate(stems_cut):
        for pi, p in enumerate(pos_tags_cut):
            for ai, a in enumerate(afsets_cut):
                results[(s, p, a)] = ((corr[si, pi, ai] / corr_Z) if (corr_Z != 0.0) else 0.0)
    #                 print(f'>>Corr: {s}-{p}-{a} ==> {corr[si,pi,ai]/corr.sum()}')

    results_list = [(k, v) for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)]

    ret_stems = [s for (s, p, a), prob in results_list]
    ret_pos_tags = [p for (s, p, a), prob in results_list]
    ret_afsets = [a for (s, p, a), prob in results_list]
    ret_affixes = [afset_own_affixes[a][0] for (s, p, a), prob in results_list]
    ret_non_afset_affixes_stats_list = [afset_own_affixes[a][1] for (s, p, a), prob in results_list]
    ret_probs = [(prob * wt_prob) for (s, p, a), prob in results_list]

    # surface_forms = [pos_tag_view(p)+'/'+afset_view(a,all_afsets)+'/'+stems_vocab[s]+'/'+'-'.join([affix_view(af,all_affixes) for af in ret_affixes[i]]) for i,(s,p,a) in enumerate(sorted_results.keys())]
    surface_forms = [make_surface_form(s, ret_affixes[i], stems_vocab, all_affixes, ffi, lib) for
                     i, ((s, p, a), prob) in enumerate(results_list)]

    gen_word = surface_forms[0][0] if len(surface_forms) > 0 else '<none>'
    for i, (wrd, flg) in enumerate(surface_forms):
        if flg:
            gen_word = wrd  # + ' ({})'.format(pos_tag_view(ret_pos_tags[i]))
            break

    return (gen_word,
            surface_forms,
            ret_stems,
            ret_pos_tags,
            ret_afsets,
            ret_affixes,
            ret_probs,
            ret_non_afset_affixes_stats_list)


def decode_word(top_stems, top_pos_tags, top_afsets, top_affixes,
                top_stems_prob, top_pos_tags_prob, top_afsets_prob, top_affixes_prob,
                stems_vocab, all_afsets, all_affixes, all_afsets_inverted_index,
                pos_afset_corr, pos_stem_corr, afset_stem_corr, afset_affix_slot_corr,
                ffi, lib,
                prob_cutoff=0.3, affix_prob_cutoff=0.3,
                affix_min_prob=0.3, lprob_score_delta=2.0):
    # 1. wt resolution --> morpho_wt, -1: other
    wt_list = [i for i in range(len(all_wt_abbrevs))] + [-1]
    wt_votes = [0.0 for _ in range(len(wt_list))]
    wt_vals = [-100.0 for _ in range(len(wt_list))]

    for id, p in zip(top_stems, top_stems_prob):
        if stems_vocab[id].find(':') > 0:
            t = all_wt_abbrevs_dict[stems_vocab[id].split(':')[0]]
            wt_vals[t] = max(wt_vals[t], math.log(p))
        else:
            wt_vals[-1] = max(wt_vals[-1], math.log(p))

    wt_votes = [a + b for a, b in zip(wt_votes, wt_vals)]

    wt_vals = [-100.0 for _ in range(len(wt_list))]
    for id, p in zip(top_pos_tags, top_pos_tags_prob):
        if (id >= 5) and ((id - 5) < len(all_pos_tags)):
            nm = all_pos_tags[id - 5]["name"]
            if nm in all_wt_abbrevs_dict:
                t = all_wt_abbrevs_dict[nm]
                wt_vals[t] = max(wt_vals[t], math.log(p))
            else:
                wt_vals[-1] = max(wt_vals[-1], math.log(p))
        else:
            wt_vals[-1] = max(wt_vals[-1], math.log(p))

    wt_votes = [a + b for a, b in zip(wt_votes, wt_vals)]

    wt_vals = [-100.0 for _ in range(len(wt_list))]
    for id, p in zip(top_afsets, top_afsets_prob):
        af = id_to_afset(id, all_afsets)
        if af is not None:
            t = af.wt
            wt_vals[t] = max(wt_vals[t], math.log(p))
        else:
            wt_vals[-1] = max(wt_vals[-1], math.log(p))

    wt_votes = [a + b for a, b in zip(wt_votes, wt_vals)]

    wt_Z = sum([math.exp(v) for v in wt_votes])
    wt_probs = [(math.exp(v) / wt_Z) if (wt_Z != 0.0) else 0.0 for v in wt_votes]

    #     print('==> wt_votes: ', wt_votes)
    #     print('==> wt_probs: ', wt_probs)

    wt_tuples = sorted([(id, sc, pr) for (id, sc, pr) in zip(wt_list, wt_votes, wt_probs)], key=lambda x: x[1],
                       reverse=True)

    return_list = []
    prev_wt_score = wt_tuples[0][1]
    for wt, wt_score, wt_prob in wt_tuples:
        if (prev_wt_score - wt_score) > lprob_score_delta:
            break
        ret = decode_word_per_wt(top_stems, top_pos_tags, top_afsets, top_affixes,
                                 top_stems_prob, top_pos_tags_prob, top_afsets_prob, top_affixes_prob,
                                 stems_vocab, all_afsets, all_affixes, all_afsets_inverted_index,
                                 pos_afset_corr, pos_stem_corr, afset_stem_corr, afset_affix_slot_corr,
                                 ffi, lib, wt, wt_prob,
                                 prob_cutoff=prob_cutoff, affix_prob_cutoff=affix_prob_cutoff,
                                 affix_min_prob=affix_min_prob)
        return_list.append(ret)
        prev_wt_score = wt_score
    return return_list
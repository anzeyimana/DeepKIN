import argparse
from datetime import datetime

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_lines(file_name):
    f = open(file_name, 'r', encoding='utf-8')
    lines = [line.rstrip('\n') for line in f]
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    f.close()
    return lines
def write_lines(lines, file_name):
    f = open(file_name, 'w', encoding='utf-8')
    for l in lines:
        f.write(l+'\n')
    f.close()

def time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def date_now():
    return datetime.now().strftime("%Y-%m-%d")

def normalize_kinya_text(txt, tag_dict=None, upper_first=True):
    if tag_dict is None:
        tag_dict = dict()
    tokens = (' '.join(txt.split())).split()
    idx = 0
    sent = ''
    tot = 0
    while (idx < len(tokens)):
        tok = tokens[idx]
        sep = ''
        if len(tok) > 0:
            sep = (('' if ((tok[0] in ',.;?!:') or (idx==0)) else ' ') if (len(tok) > 0) else '')
            if (idx == 1) and (sent[-1] in '’‘‘`"\'“”‘’«»“"'):
                sep = ''
            if (idx == (len(tokens)-1)) and (tok[0] in '’‘‘`"\'“”‘’«»“"'):
                sep = ''
        added = False
        if (len(tok) > 0):
            if ((tok[-1] == '\'') or ((tok[-1] == 'a') and (len(tok) < 4) and (len(tok) > 1))) and (idx<(len(tokens)-1)):
                next_tok = tokens[idx+1]
                if len(next_tok) > 0:
                    if ((next_tok[0] in 'iuoae') or (tok[-1] == '\'')):
                        curr_tag = 'UNK'
                        next_tag = 'UNK'
                        if tok in tag_dict:
                            curr_tag = tag_dict[tok]
                        if next_tok in tag_dict:
                            next_tag = tag_dict[next_tok]
                        if (curr_tag != 'V') and (curr_tag != 'N') and (curr_tag != 'DE') and (not ((curr_tag == 'PO') and (next_tag == 'V'))):
                            sent += sep + tok[:-1] + '\'' + next_tok
                            idx += 2
                            added = True
                            tot += 1
        if (not added) and (idx > 1) and (len(sent) > 0) and (len(tok) > 0):
            if (tok[0] in '0123456789') and (sent[-1] in ',.'):
                if len(sent) > 1:
                    if sent[-2] in '0123456789':
                        sent += tok
                        idx += 1
                        added = True
                        tot += 1
        if (not added) and (idx > 1) and (len(sent) > 0) and (len(tok) > 0):
            if (tok == '%') and (sent[-1] in '0123456789'):
                sent += tok
                idx += 1
                added = True
                tot += 1
        if (not added) and (idx > 0) and (len(sent) > 0) and (len(tok) > 0) and ((idx+1) < len(tokens)):
            if ((tok == '-') or (tok == '–') or (tok == '_')) and (tokens[idx-1].isalnum()) and (tokens[idx+1].isalnum()):
                sent += tok+tokens[idx+1]
                idx += 2
                added = True
                tot += 2
        if not added:
            sent += sep + tok
            idx += 1
            tot += 1
    if tot > 1:
        if upper_first:
            sent = sent[:1].upper()+sent[1:]
    if len(sent) > 0:
        sent = sent.replace('<unk>', ' ')
        sent = ' '.join(sent.replace('\n',' ').replace('\t',' ').split())
    return sent

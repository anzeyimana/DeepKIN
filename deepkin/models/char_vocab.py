import unicodedata
import re
from typing import List

PAD_ID = 0
UNK_ID = 1
MSK_ID = 2
BOS_ID = 3
EOS_ID = 4

NUM_SPECIAL_TOKENS = 5

VOCAB_TOKENS = ['<pad>', '<unk>', '<mask>', '<s>', '</s>', '|', '~', 'i', 'u', 'o', 'a', 'e', 'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'r', 'l', 's', 't', 'v', 'y', 'w', 'z', 'x', 'q', ',', '.', '?', '!', '-', ':', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '\'']

SIL_ID = 5
BLANK_ID = 6

KINSPEAK_VOCAB = {v:i for i,v in enumerate(VOCAB_TOKENS)}
KINSPEAK_VOCAB_IDX = {i:v for i,v in enumerate(VOCAB_TOKENS)}

def normalize_text(string, encoding="utf-8") -> str:
    string = string.decode(encoding) if isinstance(string, type(b'')) else string
    string = string.replace('`','\'')
    string = string.replace("'", "\'")
    string = string.replace("‘", "\'")
    string = string.replace("’", "\'")
    string = string.replace("‚", "\'")
    string = string.replace("‛", "\'")
    string = string.replace(u"æ", u"ae").replace(u"Æ", u"AE")
    string = string.replace(u"œ", u"oe").replace(u"Œ", u"OE")
    return unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode("ascii").lower()

def text_to_id_sequence(text) -> List[int]:
    txt = normalize_text(text)
    txt = re.sub(r"\s+", '|', txt).strip()
    return [KINSPEAK_VOCAB[t] for t in txt if t in KINSPEAK_VOCAB]

def id_sequence_to_text(seq) -> str:
    return ''.join([' ' if (id == SIL_ID) else KINSPEAK_VOCAB_IDX[id] for id in seq if id!=BLANK_ID])

def syllbe_vocab_size() -> int:
    return len(VOCAB_TOKENS)

def syllbe_vocab_units() -> List[str]:
    return VOCAB_TOKENS

if __name__ == '__main__':
    assert len(VOCAB_TOKENS) == len(KINSPEAK_VOCAB), "Vocab length mismatch"
    assert len(VOCAB_TOKENS) == len(KINSPEAK_VOCAB_IDX), "Vocab length mismatch"

    tests = ['abana', 'terimbere',
             'Miel ok',
             'Gourmand grande ?',
             '5 Xaviersss wacu twese',
             'Ishyano riratugwiriye ',
             "Nta munsi w’ubusa wigeze wirenza muri uyu mwaka Abanyarwanda batibajije ku myitwarire ya Perezida wa RDC, kubera uburyo batunguwe mu kanya nk’ako guhumbya."]
    for test in tests:
        seq = text_to_id_sequence(test)
        print(test, [KINSPEAK_VOCAB_IDX[id] for id in seq], id_sequence_to_text(seq))
    file = "/mnt/NVM/KINLP/data/kinspeak/jw_speech/samples/new_txt/77327059-53be-42ee-8ea1-21e1b2c30999_299_203471_209245_3171_3273.txt"
    f = open(file, 'r')
    test = f.read()
    f.close()
    seq = text_to_id_sequence(test)
    print(test, '\n', [KINSPEAK_VOCAB_IDX[id] for id in seq], '\n', id_sequence_to_text(seq))
    print()

    text = "Igirire ikizere kandi uhore wisanzuye Abagore benshi banezezwa no kubona umugabo wihagazeho kandi wifitiye ikizere n' ubwo bamwe muri bo bigira nk' aho bitabafashe ho ."
    print(id_sequence_to_text(text_to_id_sequence(text)))

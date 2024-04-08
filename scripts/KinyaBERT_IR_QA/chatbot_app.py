import gc
import sys
import time
import progressbar

import numpy as np
import torch
from numpy.linalg import norm
from torch.nn.utils.rnn import pad_sequence

from deepkin.clib.libkinlp.kinlpy import build_kinlpy_lib, parse_text_to_morpho_sentence
from deepkin.models.arguments import py_trainer_args
from deepkin.models.kinyabert import KinyaBERT_PretrainModel, KinyaBERT_SequenceClassifier
from deepkin.models.modules import BaseConfig
from mlm_data import prepare_input_segments


def get_sentence_embedding(ffi, lib, model:KinyaBERT_PretrainModel, device, text):
    model.eval()
    sentence = parse_text_to_morpho_sentence(ffi, lib, text)
    (lm_morphs, pos_tags, affixes, tokens_lengths, stems) = prepare_input_segments([sentence])
    tokens_lengths = tokens_lengths
    input_sequence_lengths = [len(tokens_lengths)]
    lm_morphs = torch.tensor(lm_morphs).to(device)
    pos_tags = torch.tensor(pos_tags).to(device)
    affixes = torch.tensor(affixes).to(device)
    stems = torch.tensor(stems).to(device)
    afx = affixes.split(tokens_lengths)
    afx_padded = pad_sequence(afx, batch_first=False)
    afx_padded = afx_padded.to(dtype=torch.long).to(device)
    m_masks = [torch.zeros((x + 4), dtype=torch.bool) for x in tokens_lengths]
    m_masks_padded = pad_sequence(m_masks, batch_first=True, padding_value=1).to(device)  # Shape: (L, 4+M)
    masks = [torch.zeros(x, dtype=torch.bool) for x in input_sequence_lengths]
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=1).to(device) # Shape: N x S
    with torch.no_grad():
        bert_output = model.encoder(lm_morphs, pos_tags, stems, input_sequence_lengths,
                                    afx_padded, m_masks_padded, masks_padded) # Shape: L x N x E, with L = max sequence length
    return bert_output[0,0,:].detach().cpu().numpy()


def answer_question(question, QA_DATA, ffi, lib, model, device):
    Q = get_sentence_embedding(ffi, lib, model, device, question)
    max_sim = -999999.0
    max_answers = []
    for it_embeds,responses in QA_DATA:
        sim = []
        for A in it_embeds:
            sim.append(float(np.dot(Q,A)/(norm(Q)*norm(A))))
        similarity = sum(sim)/len(sim)
        if similarity > max_sim:
            max_sim = similarity
            max_answers = responses
    return max_answers, max_sim

if __name__ == '__main__':
    import json
    build_kinlpy_lib()
    from kinlpy import ffi, lib
    lib.init_kinlp_socket()

    rank = 0
    device = torch.device('cuda:%d' % rank)
    cfg = BaseConfig()
    args = py_trainer_args(silent=True)

    model = KinyaBERT_PretrainModel(args, cfg)
    model.float()
    print(f'Loading KinyaBERT model from {args.pretrained_model_file} ...')
    kb_state_dict = torch.load(args.pretrained_model_file, map_location='cpu')
    model.load_state_dict(kb_state_dict['model_state_dict'])
    model.eval()
    del kb_state_dict
    gc.collect()
    model = model.to(device)

    print('Parsing intents data ...')
    INTENTS_FILE = args.dev_unparsed_corpus
    with open(INTENTS_FILE, 'r', encoding='utf-8') as read_file:
        intent_data = json.load(read_file)

    QA_DATA = []
    with progressbar.ProgressBar(max_value=len(intent_data['intents']), redirect_stdout=True) as bar:
        for itr,it in enumerate(intent_data['intents']):
            bar.update(itr)
            it_embeds = []
            for pattern in it['patterns']:
                if len(pattern) >= 10:
                    try:
                        embedding = get_sentence_embedding(ffi, lib, model, device, pattern)
                        it_embeds.append(embedding)
                    except:
                        pass
            if (len(it_embeds) > 0) and (len(it['responses'][0]) > 1):
                QA_DATA.append((it_embeds,it['responses']))
    print(f'Got {len(QA_DATA)} embedded intents!')

    while True:
        question = input("\nEnter your question(e/E/q/Q to quit): ")
        if question in {'exit', 'EXIT', 'e', 'E', 'quit', 'QUIT', 'q', 'Q'}:
            print('Exiting ...')
            sys.exit(0)
        t0 = time.perf_counter()
        max_answers, max_sim = answer_question(question, QA_DATA, ffi, lib, model, device)
        t1 = time.perf_counter()
        print(f'\n{max_sim:.3f} : {max_answers[0]}')

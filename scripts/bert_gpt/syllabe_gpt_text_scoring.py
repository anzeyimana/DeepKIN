import warnings
warnings.filterwarnings("ignore")

import os
import sys
from typing import List

import torch

from deepkin.models.arguments import py_trainer_args
from deepkin.models.syllabe_gpt_model import SyllabeGPT
from deepkin.models.syllabe_vocab import BOS_ID, text_to_id_sequence
from deepkin.models.util import generate_input_key_padding_mask, generate_square_subsequent_mask

def batch_lm_scores(kinya_sentences: List[str], syllabe_gpt: SyllabeGPT, device:torch.device) -> List[float]:
    syllabe_ids = []
    syllabe_id_lengths = []
    for sentence in kinya_sentences:
        sent_ids = [BOS_ID] + text_to_id_sequence(sentence)
        syllabe_ids.extend(sent_ids)
        syllabe_id_lengths.append(len(sent_ids))

    syllabe_ids = torch.tensor(syllabe_ids).to(device)
    with torch.no_grad():
        syllabe_gpt.eval()
        tgt_key_padding_mask = generate_input_key_padding_mask(syllabe_id_lengths, ignore_last=True).to(syllabe_ids.device)
        tgt_decoder_mask = generate_square_subsequent_mask(max(syllabe_id_lengths)).to(syllabe_ids.device)
        lm_scores = syllabe_gpt.batched_nll_losses(syllabe_ids, syllabe_id_lengths, tgt_key_padding_mask, tgt_decoder_mask)
    return lm_scores

if __name__ == '__main__':
    # Interactive LM scoring with SyllabeGPT

    # 1. Setup device
    rank = 0
    device = torch.device('cuda:%d' % rank)

    # 2. Setup SyllabeGPT model
    args = py_trainer_args(list_args=[], silent=True)
    syllabe_gpt = SyllabeGPT(max_seq_len=args.syllabe_max_seq_len).to(device)
    syllabe_gpt.float()
    KINLP_HOME = (os.environ['KINLP_HOME']) if ('KINLP_HOME' in os.environ) else ('/opt/KINLP')
    home_path = KINLP_HOME + "/"
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    state_dict = torch.load(home_path + "models/syllabe_gpt_final_best_valid_2023-01-05_base_2022-12-30.pt", map_location=map_location)
    syllabe_gpt.load_state_dict(state_dict['model_state_dict'])
    del state_dict
    syllabe_gpt.eval()

    # 3. LM scoring
    while True:
        input_sentence = input("\nInput Kinyarwanda text(e/E/q/Q to Exit): ")
        if input_sentence in {'exit', 'EXIT', 'e', 'E', 'quit', 'QUIT', 'q', 'Q'}:
            print('Exiting ...')
            sys.exit(0)
        scores = batch_lm_scores([input_sentence], syllabe_gpt, device)
        print(f'LM Score: {scores[0]:.6f}\n')

import os
import sys
from argparse import ArgumentParser
import torch
import time
import math
import progressbar
from deepkin.models.arguments import py_trainer_args
from deepkin.models.syllabe_gpt_model import SyllabeGPT
from deepkin.models.syllabe_vocab import BOS_ID, text_to_id_sequence, EOS_ID, id_sequence_to_text
from deepkin.utils.misc_functions import normalize_kinya_text


def beam_search_lm_inference(seed_sentence, syllabe_gpt: SyllabeGPT, beam_size=20, max_length=512):
    seed_ids = [BOS_ID] + text_to_id_sequence(seed_sentence)
    pending = [(seed_ids, 0)]
    complete = []
    seed_len = len(seed_ids)

    with progressbar.ProgressBar(max_value=max_length, redirect_stdout=True) as bar:
        for itr in range(max_length):
            bar.update(itr)
            syllabe_ids = torch.tensor([t for tl in pending for t in tl[0]], device=device, dtype=torch.long)
            syllabe_id_lengths = [len(tl[0]) for tl in pending]
            _, next_syllabe_probs = syllabe_gpt.predict(syllabe_ids, syllabe_id_lengths)  # (N, |C|)
            probs, preds = torch.topk(next_syllabe_probs, beam_size, dim=1)
            preds = preds.tolist()
            probs = [[p for p in pl] for pl in probs.tolist()]
            expansion = [((pend[0] + [n]), (pend[1] + p)) for pend, next, next_prob in zip(pending, preds, probs) for
                         n, p in zip(next, next_prob)]
            complete = complete + [(t, p) for (t, p) in expansion if t == EOS_ID]
            pending = [(t, p) for (t, p) in expansion if t != EOS_ID]
            pending = sorted(pending, key=lambda x: x[1], reverse=True)
            pending = pending[:beam_size]

    alpha = 0.2
    complete_text = [(id_sequence_to_text(t[seed_len:]), (p / math.pow(((len(t) + 5.0) / 6.0), alpha))) for (t, p) in
                     complete]
    pending_text = [(id_sequence_to_text(t[seed_len:]), (p / math.pow(((len(t) + 5.0) / 6.0), alpha))) for (t, p) in
                    pending]

    complete_text = sorted(complete_text, key=lambda x: x[1], reverse=True)
    pending_text = sorted(pending_text, key=lambda x: x[1], reverse=True)

    return pending_text, complete_text

if __name__ == '__main__':
    parser = ArgumentParser(description="GPT Inference arguments")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--max-output-length", type=int, default=180)
    local_arguments = parser.parse_args()
    seed_sentence = local_arguments.input

    rank = 0
    device = torch.device('cuda:%d' % rank)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    lars = []
    args = py_trainer_args(list_args=lars, silent=True)
    syllabe_gpt = SyllabeGPT(max_seq_len=args.syllabe_max_seq_len).to(device)
    syllabe_gpt.float()
    KINLP_HOME = (os.environ['KINLP_HOME']) if ('KINLP_HOME' in os.environ) else ('/opt/KINLP')
    home_path = KINLP_HOME + "/"
    state_dict = torch.load(home_path + "models/syllabe_gpt_final_best_valid_2023-01-05_base_2022-12-30.pt", map_location=map_location)
    syllabe_gpt.load_state_dict(state_dict['model_state_dict'])
    del state_dict
    syllabe_gpt.eval()

    beam_size = 4
    max_length = local_arguments.max_output_length

    start = time.time()
    pending_texts, complete_texts = beam_search_lm_inference(seed_sentence, syllabe_gpt, beam_size=beam_size, max_length=max_length)
    end = time.time()

    while True:
        print('\n\nInput: {}\n'.format(seed_sentence))
        print('\nSyllabeGPT: PENDING:>> after {:.2f} seconds:\n'.format(end - start))
        for num,(sentence, new_sentence_prob) in enumerate(pending_texts):
            print('{}. {:.4f}'.format(num+1, new_sentence_prob), '\t... ', normalize_kinya_text(sentence, upper_first=False))

        if len(complete_texts) > 0:
            print('\nSyllabeGPT COMPLETED:>>\n')
            for num, (sentence, new_sentence_prob) in enumerate(complete_texts):
                print('{}. {:.4f}'.format(num+1, new_sentence_prob), '\t... ', normalize_kinya_text(sentence, upper_first=False))
        seed_sentence = input("\nInput more text(e/E/q/Q to Exit): ")
        if seed_sentence in {'exit', 'EXIT', 'e', 'E', 'quit', 'QUIT', 'q', 'Q'}:
            print('Exiting ...')
            sys.exit(0)
        start = time.time()
        pending_texts, complete_texts = beam_search_lm_inference(seed_sentence, syllabe_gpt, beam_size=beam_size, max_length=max_length)
        end = time.time()

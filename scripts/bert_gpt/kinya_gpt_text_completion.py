from __future__ import print_function, division

import sys
from argparse import ArgumentParser

from deepkin.models.kinyagpt_decode import gpt_init_decode_model, gpt_auto_complete

if __name__ == '__main__':
    import time
    parser = ArgumentParser(description="GPT Inference arguments")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--max-output-length", type=int, default=30)
    local_arguments = parser.parse_args()
    seed_sentence = local_arguments.input

    max_inference_table_length = 8
    beam_size = 4
    max_text_length = local_arguments.max_output_length
    max_completed = 10

    model_setup = gpt_init_decode_model()

    t0 = time.perf_counter()
    complete, pending = gpt_auto_complete(model_setup, seed_sentence, max_text_length,
                                          max_completed,
                                          max_morpho_inference_table_length=max_inference_table_length,
                                          max_batch_size=beam_size,
                                          show_progress=True,
                                          upper_first=False)
    t1 = time.perf_counter()

    while True:
        print('\n\nInput: {}\n'.format(seed_sentence))
        print('\nKinyaGPT: PENDING:>> after {:.2f} seconds:\n'.format(t1 - t0))
        if len(pending) > 0:
            for num, (prob, text) in enumerate(pending):
                print('{}. {:.4f}'.format(num + 1, prob), '\t... ', text)

        if len(complete) > 0:
            print('\nKinyaGPT COMPLETED:>>\n')
            for num, (prob, text) in enumerate(complete):
                print('{}. {:.4f}'.format(num + 1, prob), '\t... ', text)

        seed_sentence = input("\nInput more text(e/E/q/Q to quit): ")
        if seed_sentence in {'exit', 'EXIT', 'e', 'E', 'quit', 'QUIT', 'q', 'Q'}:
            print('Exiting ...')
            sys.exit(0)
        t0 = time.perf_counter()
        complete, pending = gpt_auto_complete(model_setup, seed_sentence, max_text_length,
                                              max_completed,
                                              max_morpho_inference_table_length=max_inference_table_length,
                                              max_batch_size=beam_size,
                                              show_progress=True,
                                              upper_first=False)
        t1 = time.perf_counter()

import progressbar

from deepkin.clib.libkinlp.kinlpy import parse_text_to_morpho_sentence
from deepkin.utils.misc_functions import time_now
from scripts.data.preprocess_ner_data import handle_appostrophe


def parse_ner_v2_dataset(ffi, lib, in_file_path, out_file_path):
    print(time_now(), 'Processing', in_file_path, '...')

    f = open(in_file_path, 'r')
    Lines = [line.rstrip('\n') for line in f]
    Lines = ['"' if (l=='""""') else l for l in Lines]
    Lines = [l.replace('“','"').replace('‘‘','"').replace('’’','"').replace('’','\'').replace('‘','\'') for l in Lines]
    f.close()

    doc_idx = [i for i in range(len(Lines)) if (len(Lines[i]) == 0)]
    if doc_idx[-1] < (len(Lines)-1):
        doc_idx.append(len(Lines))
    start_idx = 0

    text_file = open(out_file_path+'_plain.txt', 'w')
    label_file = open(out_file_path+'_labels.txt', 'w')

    parsed_file = open(out_file_path+'_parsed.txt', 'w')

    all_docs = len(doc_idx)
    with progressbar.ProgressBar(max_value=(all_docs+10), redirect_stdout=True) as bar:
        for iter,end_idx in enumerate(doc_idx):
            if (iter % 100) == 0:
                bar.update(iter)
            lines_batch = Lines[start_idx:end_idx]
            start_idx = end_idx + 1
            if (len(lines_batch) > 1):
                pwords = [l.split()[0] for l in lines_batch]
                plabels = [l.split()[1] for l in lines_batch]
                assert len(pwords)==len(plabels), "Mismatch between words: {} and labels: {}.".format(len(words), len(labels))
                words = []
                labels = []
                for (w,l) in zip(pwords, plabels):
                    ww,ll = handle_appostrophe(w, l)
                    words.extend(ww)
                    labels.extend(ll)
                sentence = ' '.join(words)
                parsed_tokens = parse_text_to_morpho_sentence(ffi, lib, sentence).tokens
                assert (len(words) <= len(parsed_tokens)), "Mismatch words: {} {} tokens: {} {} @\n{}\npw: {},\npl: {}".format(len(words), words, len(parsed_tokens), [p.raw_surface_form for p in parsed_tokens], sentence, pwords, plabels)
                start = 0
                new_labels = []
                new_parsed_tokens = []
                for wrd,lbl in zip(words,labels):
                    end = start
                    token = ''
                    while end < len(parsed_tokens):
                        token += parsed_tokens[end].raw_surface_form
                        end += 1
                        if wrd == token:
                            if (end-start) > 1:
                                pts = parsed_tokens[start:end]
                                if (pts[0].is_apostrophed != 0) and lbl.startswith('B-'):
                                    new_labels.extend(['O', lbl])
                                    new_parsed_tokens.extend(pts[:2])
                                    if (len(pts) > 2):
                                        nlb = 'I'+lbl[1:]
                                        new_labels.extend([nlb] * (len(pts)-2))
                                        new_parsed_tokens.extend(pts[2:])
                                else:
                                    new_labels.extend([lbl] * len(pts))
                                    new_parsed_tokens.extend(pts)
                            else:
                                new_labels.append(lbl)
                                new_parsed_tokens.append(parsed_tokens[start])
                            start = end
                            break
                    # Check match
                    if not ((wrd == token) and (start == end)):
                        print('Error with:',sentence)
                        print('Got parse::',' '.join([p.raw_surface_form for p in parsed_tokens]))
                        print('Mismatch: ','words: \'{}\' ~ \'{}\''.format(wrd,token), 'start:' + str(start) + ' end: ' + str(end))
                        break
                # Now assemble new dataset
                parsed_file.write(';'.join([pt.to_parsed_format() for pt in new_parsed_tokens]) + "\n")
                text_file.write(' '.join([pt.raw_surface_form for pt in new_parsed_tokens]) + "\n")
                label_file.write(' '.join([lbl for lbl in new_labels]) + "\n")

    parsed_file.close()
    text_file.close()
    label_file.close()

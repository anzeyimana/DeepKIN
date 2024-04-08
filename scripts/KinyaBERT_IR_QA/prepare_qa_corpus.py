import json
from deepkin.utils.misc_functions import normalize_kinya_text

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


if __name__ == '__main__':
    final_sentences = []
    INTENTS_FILE = '/root/qa_data.json'
    OUTPUT_CORPUS_FILE = '/root/qa_corpus.txt'
    with open(INTENTS_FILE, 'r', encoding='utf-8') as read_file:
        data = json.load(read_file)
        for it in data['intents']:
            for line in (it['patterns'] + it['responses']):
                if len(line) > 10:
                    txt = normalize_kinya_text(line)
                    if len(txt.split(' ')) > 4:
                        if txt[0] == '"' and txt[-1] == '"':
                            txt = txt[1:-1]
                        final_sentences.append(txt)
    QNLI = ['/root/DeepKIN/datasets/GLUE/QNLI/rw_translated/qnli_input_dev_input0_rw_translations.txt',
             '/root/DeepKIN/datasets/GLUE/QNLI/rw_translated/qnli_input_dev_input1_rw_translations.txt',
             '/root/DeepKIN/datasets/GLUE/QNLI/rw_translated/qnli_input_test_input0_rw_translations.txt',
             '/root/DeepKIN/datasets/GLUE/QNLI/rw_translated/qnli_input_test_input1_rw_translations.txt',
             '/root/DeepKIN/datasets/GLUE/QNLI/rw_translated/qnli_input_train_input0_rw_translations.txt',
             '/root/DeepKIN/datasets/GLUE/QNLI/rw_translated/qnli_input_train_input1_rw_translations.txt']
    for qfile in QNLI:
        with open(qfile, 'r', encoding='utf-8') as read_file:
            for line in read_file:
                if len(line) > 10:
                    txt = normalize_kinya_text(line)
                    if len(txt.split(' ')) > 4:
                        if txt[0] == '"' and txt[-1] == '"':
                            txt = txt[1:-1]
                        final_sentences.append(txt)
    final_sentences = list(set(final_sentences))
    print(f'Got {len(final_sentences)} sentences!')
    with open(OUTPUT_CORPUS_FILE, 'w', encoding='utf-8') as write_file:
        for sent in final_sentences:
            write_file.write(sent+'\n')



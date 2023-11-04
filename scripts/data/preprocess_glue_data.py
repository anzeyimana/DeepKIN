import progressbar

from deepkin.clib.libkinlp.kinlpy import parse_text_to_morpho_sentence
from deepkin.utils.misc_functions import read_lines, time_now

def process_kinya_sentences(ffi, lib, in_file_path, out_file_path):
    print(time_now(), "Processing", in_file_path, "...", flush=True)
    Lines = read_lines(in_file_path)
    parsed_file = open(out_file_path+'_parsed.txt', 'w')
    with progressbar.ProgressBar(max_value=(len(Lines)), redirect_stdout=True) as bar:
        for iter in range(len(Lines)):
            if (iter % 100) == 0:
                bar.update(iter)
            parsed_file.write(parse_text_to_morpho_sentence(ffi, lib, Lines[iter]).to_parsed_format() + "\n")
    parsed_file.close()

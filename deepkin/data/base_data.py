
def read_corpus(fn):
    f = open(fn, 'r+')
    corpus_lines = [line.rstrip('\n') for line in f]
    f.close()
    if len(corpus_lines[-1])>0:
        corpus_lines.append("")
    return corpus_lines

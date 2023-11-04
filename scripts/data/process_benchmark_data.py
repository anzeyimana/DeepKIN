from preprocess_ner_data import parse_ner_dataset
from preprocess_ner_v2_data import parse_ner_v2_dataset
from preprocess_glue_data import process_kinya_sentences
from preprocess_afrisenti_data import process_afrisenti_sentences

if __name__ == '__main__':
    from deepkin.clib.libkinlp.kinlpy import build_kinlpy_lib
    build_kinlpy_lib()
    from kinlpy import ffi, lib
    lib.init_kinlp_socket()
    print('KINLPY Lib Ready!', flush=True)

    DEEPKIN_DIR = "/root/DeepKIN/"

    # # NER V1 -------------------------------------------------------------------------------------------------------------------
    parse_ner_dataset(ffi, lib,
                       DEEPKIN_DIR + "datasets/NER/original/dev.txt",
                       DEEPKIN_DIR + "datasets/NER/parsed/dev")
    parse_ner_dataset(ffi, lib,
                       DEEPKIN_DIR + "datasets/NER/original/test.txt",
                       DEEPKIN_DIR + "datasets/NER/parsed/test")
    parse_ner_dataset(ffi, lib,
                       DEEPKIN_DIR + "datasets/NER/original/train.txt",
                       DEEPKIN_DIR + "datasets/NER/parsed/train")

    # # NER V2 -------------------------------------------------------------------------------------------------------------------
    parse_ner_v2_dataset(ffi, lib,
                       DEEPKIN_DIR + "datasets/NER_V2/original/dev.txt",
                       DEEPKIN_DIR + "datasets/NER_V2/parsed/dev")
    parse_ner_v2_dataset(ffi, lib,
                       DEEPKIN_DIR + "datasets/NER_V2/original/test.txt",
                       DEEPKIN_DIR + "datasets/NER_V2/parsed/test")
    parse_ner_v2_dataset(ffi, lib,
                       DEEPKIN_DIR + "datasets/NER_V2/original/train.txt",
                       DEEPKIN_DIR + "datasets/NER_V2/parsed/train")

    # # AFRISENTI -------------------------------------------------------------------------------------------------------------------

    process_afrisenti_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/AFRISENT/original/kr_train.tsv",
                           DEEPKIN_DIR + "datasets/AFRISENT/parsed/train_input0")
    process_afrisenti_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/AFRISENT/original/kr_dev_gold_label.tsv",
                           DEEPKIN_DIR + "datasets/AFRISENT/parsed/dev_input0")
    process_afrisenti_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/AFRISENT/original/kr_test_participants.tsv",
                           DEEPKIN_DIR + "datasets/AFRISENT/parsed/test_input0")

    # # MRPC -------------------------------------------------------------------------------------------------------------------

    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/MRPC/rw_translated/mrpc_input_dev_input0_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/MRPC/parsed/dev_input0")
    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/MRPC/rw_translated/mrpc_input_dev_input1_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/MRPC/parsed/dev_input1")

    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/MRPC/rw_translated/mrpc_input_test_input0_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/MRPC/parsed/test_input0")
    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/MRPC/rw_translated/mrpc_input_test_input1_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/MRPC/parsed/test_input1")

    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/MRPC/rw_translated/mrpc_input_train_input0_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/MRPC/parsed/train_input0")
    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/MRPC/rw_translated/mrpc_input_train_input1_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/MRPC/parsed/train_input1")

    # # RTE -------------------------------------------------------------------------------------------------------------------

    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/RTE/rw_translated/rte_input_dev_input0_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/RTE/parsed/dev_input0")
    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/RTE/rw_translated/rte_input_dev_input1_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/RTE/parsed/dev_input1")

    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/RTE/rw_translated/rte_input_test_input0_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/RTE/parsed/test_input0")
    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/RTE/rw_translated/rte_input_test_input1_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/RTE/parsed/test_input1")

    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/RTE/rw_translated/rte_input_train_input0_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/RTE/parsed/train_input0")
    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/RTE/rw_translated/rte_input_train_input1_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/RTE/parsed/train_input1")

    # # STS-B -------------------------------------------------------------------------------------------------------------------

    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/STS-B/rw_translated/stsb_input_dev_input0_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/STS-B/parsed/dev_input0")
    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/STS-B/rw_translated/stsb_input_dev_input1_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/STS-B/parsed/dev_input1")

    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/STS-B/rw_translated/stsb_input_test_input0_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/STS-B/parsed/test_input0")
    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/STS-B/rw_translated/stsb_input_test_input1_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/STS-B/parsed/test_input1")

    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/STS-B/rw_translated/stsb_input_train_input0_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/STS-B/parsed/train_input0")
    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/STS-B/rw_translated/stsb_input_train_input1_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/STS-B/parsed/train_input1")

    # # SST-2 -------------------------------------------------------------------------------------------------------------------

    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/SST-2/rw_translated/sst2_input_dev_input0_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/SST-2/parsed/dev_input0")

    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/SST-2/rw_translated/sst2_input_test_input0_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/SST-2/parsed/test_input0")

    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/SST-2/rw_translated/sst2_input_train_input0_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/SST-2/parsed/train_input0")

    # # QNLI -------------------------------------------------------------------------------------------------------------------

    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/QNLI/rw_translated/qnli_input_dev_input0_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/QNLI/parsed/dev_input0")
    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/QNLI/rw_translated/qnli_input_dev_input1_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/QNLI/parsed/dev_input1")

    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/QNLI/rw_translated/qnli_input_test_input0_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/QNLI/parsed/test_input0")
    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/QNLI/rw_translated/qnli_input_test_input1_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/QNLI/parsed/test_input1")

    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/QNLI/rw_translated/qnli_input_train_input0_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/QNLI/parsed/train_input0")
    process_kinya_sentences(ffi, lib,
                           DEEPKIN_DIR + "datasets/GLUE/QNLI/rw_translated/qnli_input_train_input1_rw_translations.txt",
                           DEEPKIN_DIR + "datasets/GLUE/QNLI/parsed/train_input1")

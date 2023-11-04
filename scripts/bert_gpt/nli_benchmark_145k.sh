#!/bin/bash
if [ x"${KINLP_HOME}" == "x" ]; then
  export KINLP_HOME="/opt/KINLP"
fi
UBUNTU_VERSION=$(lsb_release -r --short)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KINLP_HOME/lib/$UBUNTU_VERSION

TODAY=$(date '+%Y-%m-%d')

for iter_val in 0 1 2 3 4 5 6 7 8 9
do

python3 deepkin/models/tag_train_eval_model.py -g 1 --batch-size=16 --load-saved-model=false \
  --accumulation-steps=1 --model-keyword="NER_BERT_BASE_145K_$iter_val" \
  --peak-lr=0.00002 --wd=0.05 --morpho-dropout=0.1 --main-sequence-encoder-dropout=0.1 --pooler-dropout=0.2 --num-epochs=30 \
  --pretrained-model-file="$KINLP_HOME/models/kinyabert_base_2023-06-06.pt_160K.pt" \
  --devbest-cls-model-save-file-path="$KINLP_HOME/models/dev_best_kinyabert_base_NER_$TODAY.pt" \
  --final-cls-model-save-file-path="$KINLP_HOME/models/final_kinyabert_base_NER_$TODAY.pt"

python3 deepkin/models/cls_train_eval_model.py  -g 1 --load-saved-model=false \
    --home-path="$KINLP_HOME/" \
    --cls-labels=negative,neutral,positive \
    --cls-train-input0=datasets/AFRISENT/parsed/train_input0_parsed.txt \
    --cls-dev-input0=datasets/AFRISENT/parsed/dev_input0_parsed.txt \
    --cls-test-input0=datasets/AFRISENT/parsed/test_input0_parsed.txt \
    --cls-train-label=datasets/AFRISENT/parsed/train_label.txt \
    --cls-dev-label=datasets/AFRISENT/parsed/dev_label.txt \
    --cls-test-label=datasets/AFRISENT/parsed/test_label.txt \
    --batch-size=16 \
    --accumulation-steps=1 \
    --max-input-lines=800000 \
    --pretrained-model-file="$KINLP_HOME/models/kinyabert_base_2023-06-06.pt_145K.pt" \
    --model-keyword="AFRISENT_BERT_BASE_145K_$iter_val" \
    --peak-lr=0.00002 --wd=0.05 --morpho-dropout=0.0 --main-sequence-encoder-dropout=0.1 --pooler-dropout=0.0 --num-epochs=30 \
    --warmup-ratio=0.06 \
    --devbest-cls-model-save-file-path="$KINLP_HOME/models/dev_best_kinyabert_base_classifier_AFRISENT_$TODAY.pt" \
    --final-cls-model-save-file-path="$KINLP_HOME/models/final_kinyabert_base_classifier_AFRISENT_$TODAY.pt"

python3 deepkin/models/cls_train_eval_model.py  -g 1 --load-saved-model=false \
	--home-path="$KINLP_HOME/" \
	--cls-labels=0,1 \
	--cls-train-input0=datasets/GLUE/MRPC/parsed/train_input0_parsed.txt \
	--cls-train-input1=datasets/GLUE/MRPC/parsed/train_input1_parsed.txt \
	--cls-dev-input0=datasets/GLUE/MRPC/parsed/dev_input0_parsed.txt \
	--cls-dev-input1=datasets/GLUE/MRPC/parsed/dev_input1_parsed.txt \
	--cls-train-label=datasets/GLUE/MRPC/parsed/train_label.txt \
	--cls-dev-label=datasets/GLUE/MRPC/parsed/dev_label.txt \
	--batch-size=16 \
	--accumulation-steps=1 \
	--max-input-lines=800000 \
	--pretrained-model-file="$KINLP_HOME/models/kinyabert_base_2023-06-06.pt_145K.pt" \
  --model-keyword="MRPC_BERT_BASE_145K_$iter_val" \
	--peak-lr=0.00001 --wd=0.05 --morpho-dropout=0.0 --main-sequence-encoder-dropout=0.0 --pooler-dropout=0.0 --num-epochs=20 \
	--warmup-ratio=0.06 \
	--devbest-cls-model-save-file-path="$KINLP_HOME/models/devbest_classifier_MRPC_$TODAY.pt"  \
	--final-cls-model-save-file-path="$KINLP_HOME/models/final_classifier_MRPC_$TODAY.pt"  \
	--regression-target=0

python3 deepkin/models/cls_train_eval_model.py  -g 1 --load-saved-model=false \
	--home-path="$KINLP_HOME/" \
	--cls-train-input0=datasets/GLUE/RTE/parsed/train_input0_parsed.txt \
	--cls-train-input1=datasets/GLUE/RTE/parsed/train_input1_parsed.txt \
	--cls-dev-input0=datasets/GLUE/RTE/parsed/dev_input0_parsed.txt \
	--cls-dev-input1=datasets/GLUE/RTE/parsed/dev_input1_parsed.txt \
	--cls-train-label=datasets/GLUE/RTE/parsed/train_label.txt \
	--cls-dev-label=datasets/GLUE/RTE/parsed/dev_label.txt \
	--accumulation-steps=1 \
	--cls-labels=entailment,not_entailment \
	--batch-size=16 \
	--max-input-lines=800000 \
	--pretrained-model-file="$KINLP_HOME/models/kinyabert_base_2023-06-06.pt_145K.pt" \
  --model-keyword="RTE_BERT_BASE_145K_$iter_val" \
	--peak-lr=0.00002 --wd=0.01 --morpho-dropout=0.0 --main-sequence-encoder-dropout=0.0 --pooler-dropout=0.0 --num-epochs=20 \
	--warmup-ratio=0.06 \
	--devbest-cls-model-save-file-path="$KINLP_HOME/models/devbest_classifier_RTE_$TODAY.pt"  \
	--final-cls-model-save-file-path="$KINLP_HOME/models/final_classifier_RTE_$TODAY.pt"  \
	--regression-target=0

python3 deepkin/models/cls_train_eval_model.py  -g 1 --load-saved-model=false \
	--home-path="$KINLP_HOME/" \
	--cls-train-input0=datasets/GLUE/STS-B/parsed/train_input0_parsed.txt \
	--cls-train-input1=datasets/GLUE/STS-B/parsed/train_input1_parsed.txt \
	--cls-dev-input0=datasets/GLUE/STS-B/parsed/dev_input0_parsed.txt \
	--cls-dev-input1=datasets/GLUE/STS-B/parsed/dev_input1_parsed.txt \
	--cls-train-label=datasets/GLUE/STS-B/parsed/train_score.txt \
	--cls-dev-label=datasets/GLUE/STS-B/parsed/dev_score.txt \
	--accumulation-steps=1 \
	--cls-labels=0 \
	--batch-size=16 \
	--max-input-lines=800000 \
	--pretrained-model-file="$KINLP_HOME/models/kinyabert_base_2023-06-06.pt_145K.pt" \
  --model-keyword="STS-B_BERT_BASE_145K_$iter_val" \
	--peak-lr=0.00002 --wd=0.05 --morpho-dropout=0.0 --main-sequence-encoder-dropout=0.0 --pooler-dropout=0.0 --num-epochs=15 \
	--warmup-ratio=0.06 \
	--devbest-cls-model-save-file-path="$KINLP_HOME/models/devbest_classifier_STS-B_$TODAY.pt"  \
	--final-cls-model-save-file-path="$KINLP_HOME/models/final_classifier_STS-B_$TODAY.pt"  \
	--regression-target=1 --regression-scale-factor=5.0

python3 deepkin/models/cls_train_eval_model.py  -g 1 --load-saved-model=false \
	--home-path="$KINLP_HOME/" \
	--cls-train-input0=datasets/GLUE/SST-2/parsed/train_input0_parsed.txt \
	--cls-dev-input0=datasets/GLUE/SST-2/parsed/dev_input0_parsed.txt \
	--cls-train-label=datasets/GLUE/SST-2/parsed/train_label.txt \
	--cls-dev-label=datasets/GLUE/SST-2/parsed/dev_label.txt \
	--accumulation-steps=1 \
	--cls-labels=0,1 \
	--batch-size=32 \
	--max-input-lines=800000 \
	--pretrained-model-file="$KINLP_HOME/models/kinyabert_base_2023-06-06.pt_145K.pt" \
  --model-keyword="SST-2_BERT_BASE_145K_$iter_val" \
	--peak-lr=0.00002 --wd=0.05 --morpho-dropout=0.0 --main-sequence-encoder-dropout=0.0 --pooler-dropout=0.0 --num-epochs=15 \
	--warmup-ratio=0.06 \
	--devbest-cls-model-save-file-path="$KINLP_HOME/models/devbest_classifier_SST-2_$TODAY.pt"  \
	--final-cls-model-save-file-path="$KINLP_HOME/models/final_classifier_SST-2_$TODAY.pt"  \
	--regression-target=0

python3 deepkin/models/cls_train_eval_model.py  -g 1 --load-saved-model=false \
	--home-path="$KINLP_HOME/" \
	--cls-train-input0=datasets/GLUE/QNLI/parsed/train_input0_parsed.txt \
	--cls-train-input1=datasets/GLUE/QNLI/parsed/train_input1_parsed.txt \
	--cls-dev-input0=datasets/GLUE/QNLI/parsed/dev_input0_parsed.txt \
	--cls-dev-input1=datasets/GLUE/QNLI/parsed/dev_input1_parsed.txt \
	--cls-train-label=datasets/GLUE/QNLI/parsed/train_label.txt \
	--cls-dev-label=datasets/GLUE/QNLI/parsed/dev_label.txt \
	--accumulation-steps=1 \
	--cls-labels=entailment,not_entailment \
	--batch-size=32 \
	--max-input-lines=800000 \
	--pretrained-model-file="$KINLP_HOME/models/kinyabert_base_2023-06-06.pt_145K.pt" \
  --model-keyword="QNLI_BERT_BASE_145K_$iter_val" \
	--peak-lr=0.00002 --wd=0.05 --morpho-dropout=0.0 --main-sequence-encoder-dropout=0.0 --pooler-dropout=0.0 --num-epochs=15 \
	--warmup-ratio=0.06 \
	--devbest-cls-model-save-file-path="$KINLP_HOME/models/devbest_classifier_QNLI_$TODAY.pt"  \
	--final-cls-model-save-file-path="$KINLP_HOME/models/final_classifier_QNLI_$TODAY.pt"  \
	--regression-target=0

done

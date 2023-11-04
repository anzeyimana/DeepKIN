#!/bin/bash
if [ x"${KINLP_HOME}" == "x" ]; then
  export KINLP_HOME="/opt/KINLP"
fi
UBUNTU_VERSION=$(lsb_release -r --short)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KINLP_HOME/lib/$UBUNTU_VERSION

TODAY=$(date '+%Y-%m-%d')

for iter_val in 0 1 2 3 4 5 6 7 8 9
do

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
	--pretrained-model-file="$KINLP_HOME/models/kinyabert_base_2023-06-06.pt_160K.pt" \
  --model-keyword="RTE_CWGN_BERT_BASE_160K_$iter_val" \
	--peak-lr=0.00002 --wd=0.05 --morpho-dropout=0.0 --main-sequence-encoder-dropout=0.1 --pooler-dropout=0.2 --num-epochs=30 \
	--warmup-ratio=0.06 --ft-cwgnc=5.0 \
	--devbest-cls-model-save-file-path="$KINLP_HOME/models/devbest_classifier_RTE_CWGN_$TODAY.pt"  \
	--final-cls-model-save-file-path="$KINLP_HOME/models/final_classifier_RTE_CWGN_$TODAY.pt"  \
	--regression-target=0

done

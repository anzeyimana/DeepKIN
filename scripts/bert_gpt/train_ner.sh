#!/bin/bash
if [ x"${KINLP_HOME}" == "x" ]; then
  export KINLP_HOME="/opt/KINLP"
fi
UBUNTU_VERSION=$(lsb_release -r --short)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KINLP_HOME/lib/$UBUNTU_VERSION

TODAY=$(date '+%Y-%m-%d')

for iter_val in 0 1 2 3 4 5 6 7 8 9
do
#
#python3 deepkin/models/tag_train_eval_model.py -g 1 --batch-size=16 --load-saved-model=false \
#  --accumulation-steps=1 --model-keyword="NER_BERT_BASE_.$iter_val.md01_pd02.$TODAY" \
#  --peak-lr=0.00002 --wd=0.05 --morpho-dropout=0.1 --main-sequence-encoder-dropout=0.1 --pooler-dropout=0.2 --num-epochs=30 \
#  --pretrained-model-file="$KINLP_HOME/models/kinyabert_base_2023-06-06.pt_160K.pt" \
#  --devbest-cls-model-save-file-path="$KINLP_HOME/models/dev_best_kinyabert_base_NER_.$iter_val.md01_pd02.$TODAY.pt" \
#  --final-cls-model-save-file-path="$KINLP_HOME/models/final_kinyabert_base_NER_.$iter_val.md01_pd02.$TODAY.pt"

python3 deepkin/models/tag_train_eval_model.py -g 1 --batch-size=16 --load-saved-model=false \
  --accumulation-steps=1 --model-keyword="NER_BERT_BASE_.$iter_val.md00_pd02.$TODAY" \
  --peak-lr=0.00002 --wd=0.05 --morpho-dropout=0.0 --main-sequence-encoder-dropout=0.1 --pooler-dropout=0.2 --num-epochs=30 \
  --pretrained-model-file="$KINLP_HOME/models/kinyabert_base_2023-06-06.pt_160K.pt" \
  --devbest-cls-model-save-file-path="$KINLP_HOME/models/dev_best_kinyabert_base_NER_.$iter_val.md00_pd02.$TODAY.pt" \
  --final-cls-model-save-file-path="$KINLP_HOME/models/final_kinyabert_base_NER_.$iter_val.md00_pd02.$TODAY.pt"
#
#python3 deepkin/models/tag_train_eval_model.py -g 1 --batch-size=16 --load-saved-model=false \
#  --accumulation-steps=1 --model-keyword="NER_BERT_BASE_.$iter_val.md00_pd01.$TODAY" \
#  --peak-lr=0.00002 --wd=0.05 --morpho-dropout=0.0 --main-sequence-encoder-dropout=0.1 --pooler-dropout=0.1 --num-epochs=30 \
#  --pretrained-model-file="$KINLP_HOME/models/kinyabert_base_2023-06-06.pt_160K.pt" \
#  --devbest-cls-model-save-file-path="$KINLP_HOME/models/dev_best_kinyabert_base_NER_.$iter_val.md00_pd01.$TODAY.pt" \
#  --final-cls-model-save-file-path="$KINLP_HOME/models/final_kinyabert_base_NER_.$iter_val.md00_pd01.$TODAY.pt"
#
#python3 deepkin/models/tag_train_eval_model.py -g 1 --batch-size=16 --load-saved-model=false \
#  --accumulation-steps=1 --model-keyword="NER_BERT_BASE_.$iter_val.md00_pd00.$TODAY" \
#  --peak-lr=0.00002 --wd=0.05 --morpho-dropout=0.0 --main-sequence-encoder-dropout=0.1 --pooler-dropout=0.0 --num-epochs=30 \
#  --pretrained-model-file="$KINLP_HOME/models/kinyabert_base_2023-06-06.pt_160K.pt" \
#  --devbest-cls-model-save-file-path="$KINLP_HOME/models/dev_best_kinyabert_base_NER_.$iter_val.md00_pd00.$TODAY.pt" \
#  --final-cls-model-save-file-path="$KINLP_HOME/models/final_kinyabert_base_NER_.$iter_val.md00_pd00.$TODAY.pt"

done
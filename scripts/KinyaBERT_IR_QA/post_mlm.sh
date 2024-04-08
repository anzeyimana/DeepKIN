UBUNTU_VERSION=$(lsb_release -r --short)
export KINLP_HOME=/opt/KINLP
export PATH=$PATH:$KINLP_HOME:$KINLP_HOME/bin/$UBUNTU_VERSION
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KINLP_HOME/lib/$UBUNTU_VERSION:/usr/lib/x86_64-linux-gnu

python3 finetune_mlm_kinyabert.py \
  -g 2 \
  --bert-batch-size=16 \
  --bert-accumulation-steps=160 \
  --post-mlm-epochs=40 \
  --pretrained-model-file="/opt/KINLP/models/kinyabert_base_2023-06-06.pt_160K.pt" \
  --train-unparsed-corpus="/root/qa_corpus.txt"

#!/bin/bash
if [ x"${KINLP_HOME}" == "x" ]; then
  export KINLP_HOME="/opt/KINLP"
fi
UBUNTU_VERSION=$(lsb_release -r --short)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KINLP_HOME/lib/$UBUNTU_VERSION

python3 scripts/bert_gpt/test_mlm_inference.py  --pretrained-model-file="$KINLP_HOME/models/kinyabert_base_2023-06-06.pt_160K.pt"

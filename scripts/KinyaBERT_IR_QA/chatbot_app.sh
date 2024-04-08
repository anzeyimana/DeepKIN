UBUNTU_VERSION=$(lsb_release -r --short)
export KINLP_HOME=/opt/KINLP
export PATH=$PATH:$KINLP_HOME:$KINLP_HOME/bin/$UBUNTU_VERSION
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KINLP_HOME/lib/$UBUNTU_VERSION

python3 tantine_app.py \
  -g 1 \
  --pretrained-model-file="/opt/KINLP/models/kinyabert_post_mlm_model_DATE.pt" \
  --dev-unparsed-corpus="/root/qa_data.json"

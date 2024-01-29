UBUNTU_VERSION=$(lsb_release -r --short)
export KINLP_HOME=/opt/KINLP
export PATH=$PATH:$KINLP_HOME:$KINLP_HOME/bin/$UBUNTU_VERSION
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KINLP_HOME/lib/$UBUNTU_VERSION

mkdir -p /root/DeepKIN/scripts/classification_regression/checkpoints

python3 semeval2024_str_train.py \
  -g 1 \
  --train-dataset="/root/Semantic_Relatedness_SemEval2024/Track A/kin/kin_train.csv" \
  --valid-dataset="/root/Semantic_Relatedness_SemEval2024/Track A/kin/kin_dev_with_labels.csv" \
  --pretrained-model-file="/opt/KINLP/models/kinyabert_base_2023-06-06.pt_160K.pt" \
  --models-save-dir="/root/DeepKIN/scripts/classification_regression/checkpoints" \
  --devbest-model-file="semeval2024_kin_devbest.pt" \
  --final-model-file="semeval2024_kin_final.pt" \
  --devbest-model-id="devbest" \
  --final-model-id="final" \
  --regression-target=true \
  --regression-scale-factor=1.0 \
  --batch-task-id="1" \
  --default-device=0 \
  --from_db_app=false \
  --num-epochs=100


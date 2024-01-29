UBUNTU_VERSION=$(lsb_release -r --short)
export KINLP_HOME=/opt/KINLP
export PATH=$PATH:$KINLP_HOME:$KINLP_HOME/bin/$UBUNTU_VERSION
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KINLP_HOME/lib/$UBUNTU_VERSION

python3 semeval2024_str_eval.py \
  -g 1 \
  --eval-dataset="/root/Semantic_Relatedness_SemEval2024/Track A/kin/kin_test.csv" \
  --pretrained-model-file="/root/DeepKIN/scripts/classification_regression/checkpoints/semeval2024_kin_devbest.pt" \
  --output-file="/root/pred_kin_a.csv" \
  --regression-target=true \
  --regression-scale-factor=1.0 \
  --default-device=0 \
  --from_db_app=false \
  --num-epochs=100

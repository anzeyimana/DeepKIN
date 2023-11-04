#!/bin/bash
if [ x"${KINLP_HOME}" == "x" ]; then
  export KINLP_HOME="/opt/KINLP"
fi
UBUNTU_VERSION=$(lsb_release -r --short)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KINLP_HOME/lib/$UBUNTU_VERSION

python3 scripts/data/process_benchmark_data.py

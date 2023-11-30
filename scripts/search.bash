#!/bin/bash
# Usage:
# -- $1: the checkpoint to search
# -- $2: the output folder where the hypotheses are written to
# -- $3: the path to the PosNet root directory

topic="posnet" # ['baselines', 'posnet']
# With and without postfix "*Base", e.g. (transformerBase):
# ['transformer', 'reltransformer', 'fnet', 'gaussiannet', 'lightconv', 'linearattn', 'posnet']
model="posnetBase"
task="wmt.en-de" # ['iwslt.de-en', 'wmt.en-de', 'wmt.en-fr', 'wmt.en-zh']

number_of_gpus=1
checkpoint_path=$1
output_folder=$2
code_base=$3

export PYTHONPATH=${PYTHONPATH}:$code_base/models/$topic/
export PYTHONPATH=${PYTHONPATH}:$code_base/pytorchmt/

python3 $code_base/pytorchmt/pytorchmt/search.py \
    --config $code_base/models/$topic/configs/$model.$task.yaml \
    --checkpoint-path $checkpoint_path \
    --output-folder $output_folder \
    --number-of-gpus 1
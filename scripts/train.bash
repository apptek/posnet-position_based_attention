#!/bin/bash
# Usage:
# -- $1: the path to the output folder where checkpoints and logs are stored
# -- $2: the path to the PosNet root directory

topic="posnet" # ['baselines', 'posnet']
# With and without postfix "*Base", e.g. (transformerBase):
# ['transformer', 'reltransformer', 'fnet', 'gaussiannet', 'lightconv', 'linearattn', 'posnet']
model="posnetBase"
task="wmt.en-de" # ['iwslt.de-en', 'wmt.en-de', 'wmt.en-fr', 'wmt.en-zh']

code_base=$2
output_folder=$1
number_of_gpus=1

export PYTHONPATH=${PYTHONPATH}:$code_base/models/$topic/
export PYTHONPATH=${PYTHONPATH}:$code_base/pytorchmt/

if (( number_of_gpus == 0 )); then
    python3 $code_base/pytorchmt/pytorchmt/train.py \
        --config $code_base/models/$topic/configs/$model.$task.yaml \
        --output-folder $output_folder \
        --number-of-gpus $number_of_gpus # This overwrites the value from the .yaml
else
    mpirun -np $number_of_gpus \
        -bind-to none -map-by slot \
        -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
        -mca pml ob1 -mca btl ^openib \
        python3 $code_base/pytorchmt/pytorchmt/train.py \
            --config $code_base/models/$topic/configs/$model.$task.yaml \
            --output-folder $output_folder \
            --number-of-gpus $number_of_gpus # This overwrites the value from the .yaml
fi



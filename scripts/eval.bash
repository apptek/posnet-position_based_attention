#!/bin/bash
# Usage:
# -- $1: path to the hypotheses file
# -- $2: the path to the PosNet root directory

task="wmt.en-de" # ['iwslt.de-en', 'wmt.en-de', 'wmt.en-fr']

hyp_file=$1
code_base=$2

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

if [[ "$task" = "iwslt.de-en" ]]; then
    src_file=$code_base/data/iwslt14/de-en/detokenized/test.de.detok
    ref_file=$code_base/data/iwslt14/de-en/detokenized/test.en.detok
    language=en
elif [[ "$task" = "wmt.en-de" ]]; then
    src_file=$code_base/data/wmt14/en-de/detokenized/test.en.detok
    ref_file=$code_base/data/wmt14/en-de/detokenized/test.de.detok
    language=de
elif [[ "$task" = "wmt.en-fr" ]]; then
    src_file=$code_base/data/wmt14/en-fr/detokenized/test.en.detok
    ref_file=$code_base/data/wmt14/en-fr/detokenized/test.fr.detok
    language=fr
fi

# POSTPROCESS
sed 's:@@ ::g' $hyp_file | mosesdecoder/scripts/tokenizer/detokenizer.perl -l $language > ${hyp_file}_processed

# BLEU
bleu=$(sacrebleu $ref_file -i ${hyp_file}_processed -m bleu -b -w 4)
echo '=== BLEU: ' $bleu

# BLEURT
wget -nc https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
unzip -n BLEURT-20.zip
python3 -m bleurt.score_files \
    -candidate_file=${hyp_file}_processed \
    -reference_file=$ref_file \
    -bleurt_checkpoint=BLEURT-20 > bleurt_scores
bleurt=$(cat bleurt_scores | awk '{x+=$0}END{print x/NR}')
echo '=== BLEURT: ' $bleurt

# COMET
comet=$(comet-score -s $src_file -t ${hyp_file}_processed -r $ref_file --only_system --model Unbabel/wmt20-comet-da | grep 'score:' | cut -d " " -f 2)
echo '=== COMET: ' $comet

echo '===== SUMMARY: BLEU: ' $bleu ', BLEURT: ' $bleurt ', COMET: ' $comet
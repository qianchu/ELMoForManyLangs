#!/usr/bin/env bash

ch_vocab=$1 # absolute path for sentence files
model_path=$2
gpu=$3
batch=$4
outputf=$5
outputfname=$6
sent_max_len=$7


# conll vocab
python ../../corpora/convert_vocab2conll.py $ch_vocab $ch_vocab'.conll'

# produce elmo representation
output_prefix_ch=$outputfname'.'$(basename $model_path)

echo python -m elmoformanylangs test  --max $sent_max_len --batch $batch  --input_format conll  --gpu $gpu   --input $ch_vocab'.conll'     --model $model_path    --output_prefix $output_prefix_ch   --output_layer -1   --output_format hdf5
python -m elmoformanylangs test  --max $sent_max_len --batch $batch  --input_format conll  --gpu $gpu   --input $ch_vocab'.conll'     --model $model_path    --output_prefix $output_prefix_ch   --output_layer -1  --output_format hdf5
echo 'output file: '$output_prefix_ch'.ly'$output_layer'.hdf5'

# convert to word2vec
if [ "${outputf}" == "word2vec" ]; then
    python ./h5py2word2vec.py $output_prefix_ch'.ly'$output_layer'.hdf5' $output_prefix_ch'.ly'$output_layer'.word2vec'
fi
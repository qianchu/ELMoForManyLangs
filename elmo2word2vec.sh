#!/usr/bin/env bash

ch_vocab=$1 # absolute path for sentence files
model_path=$2
output_layer=$3
gpu=$4
batch=$5


# conll vocab
python ../../corpora/convert_vocab2conll.py $ch_vocab $ch_vocab'.conll'

# produce elmo representation
output_prefix_ch=$model_path$(basename $ch_vocab)

python -m elmoformanylangs test   --batch $batch  --input_format conll  --gpu $gpu   --input $ch_vocab'.conll'     --model $model_path    --output_prefix $model_path$(basename $output_prefix_ch)   --output_layer $output_layer   --output_format hdf5
echo 'output file: '$output_prefix_ch'.ly'$output_layer'.hdf5'

# convert to word2vec
python ./h5py2word2vec.py $output_prefix_ch'.ly'$output_layer'.hdf5' $output_prefix_ch'.ly'$output_layer'.word2vec'

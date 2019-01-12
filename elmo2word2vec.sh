#!/usr/bin/env bash

ch_vocab=$1 #absolute path for ch vocab
en_vocab=$2 #absolute path for en vocab
output_layer=$3


# conll vocab
python ../../corpora/convert_vocab2conll.py $ch_vocab $ch_vocab'.conll'
python ../../corpora/convert_vocab2conll.py $en_vocab $en_vocab'.conll'

# produce elmo representation
output_prefix_ch='./models/chinese_elmo/ch_vocab'
output_prefix_en='./models/english_elmo/en_vocab'

python -m elmoformanylangs test     --input_format conll     --input $ch_vocab'.conll'     --model ./models/chinese_elmo/    --output_prefix './models/chinese_elmo/'$(basename $output_prefix_ch)   --output_layer $output_layer   --output_format hdf5
python -m elmoformanylangs test     --input_format conll     --input $en_vocab'.conll'     --model ./models/english_elmo/    --output_prefix './models/english_elmo/'$(basename $output_prefix_en)   --output_layer $output_layer   --output_format hdf5

# convert to word2vec
python ./h5py2word2vec.py $output_prefix_ch'.ly'$output_layer'.hdf5' $output_prefix_ch'.ly'$output_layer'.word2vec'

python ./h5py2word2vec.py $output_prefix_en'.ly'$output_layer'.hdf5' $output_prefix_en'.ly'$output_layer'.word2vec'

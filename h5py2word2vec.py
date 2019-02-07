__author__ = 'qianchu_liu'
import h5py

def h5py2word2vec(h5py_file,output_f):
    with open(output_f,'w') as output_f:
        f=h5py.File(h5py_file)
        output_f.write('{0} {1}\n'.format(str(len(f.keys())),len(f[list(f.keys())[0]][0])))
        for sent in f.keys():
            for i, word in enumerate(sent.split('\t')):
                if len(sent.split('\t'))==1:
                    output_f.write('{0}'.format(sent)+' '+' '.join([str(v) for v in f[sent][i]])+'\n')
                else:
                    output_f.write('{0}||{1}'.format(sent,i)+' '+' '.join([str(v) for v in f[sent][i]])+'\n')

if __name__=='__main__':
    import sys
    hdf5file=sys.argv[1]
    output_word2vec=sys.argv[2]
    h5py2word2vec(hdf5file,output_word2vec)
    # h5py2word2vec('models/ELMoForManyLangs/chinese_elmo/ch_vocab.ly-1.hdf5','models/ELMoForManyLangs/chinese_elmo/ch_vocab.ly-1.word2vec')
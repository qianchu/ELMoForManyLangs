__author__ = 'qianchu_liu'

import h5py
import numpy as np
import codecs
from allennlp.modules.elmo import Elmo,batch_to_ids
# from allennlp.commands.elmo import ElmoEmbedder as Elmo


def show_heatmap(vectors,keys):
    import matplotlib.pyplot as plt
    # sphinx_gallery_thumbnail_number = 2

    vegetables = keys
    farmers = keys

    harvest = vectors

    fig, ax = plt.subplots()
    im = ax.imshow(harvest)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(farmers)))
    ax.set_yticks(np.arange(len(vegetables)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(farmers)
    ax.set_yticklabels(vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.show()
def matrix_norm(w):
    s = np.sqrt((w * w).sum(1))
    s[s==0.] = 1.
    w /= s.reshape((s.shape[0], 1))
    return w


def vector_heatmap(vectors):
    vectors=matrix_norm(vectors)

    matrix=np.dot(vectors,vectors.T)

    return matrix

def read_conll_corpus(path, max_chars=None):
  """
  read text in CoNLL-U format.

  :param path:
  :param max_chars:
  :return:
  """
  dataset = []
  textset = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for payload in fin.read().strip().split('\n\n'):
      data = ['<bos>']
      text = []
      lines = payload.splitlines()
      body = [line for line in lines if not line.startswith('#')]
      for line in body:
        fields = line.split('\t')
        num, token = fields[0], fields[1]
        if '-' in num or '.' in num:
          continue
        text.append(token)
        if max_chars is not None and len(token) + 2 > max_chars:
          token = token[:max_chars - 2]
        data.append(token)
      data.append('<eos>')
      dataset.append(data)
      textset.append(text)
  return dataset, textset

def elmo_large(conll_file):
    dataset,textset=read_conll_corpus(conll_file)

    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    elmo = Elmo(options_file, weight_file, 1)
    sentences = [sent[1:-1] for sent in dataset]

    print (sentences)
    character_ids = batch_to_ids(sentences)
    embeddings = elmo(character_ids)['elmo_representations'][0].detach()
    vectors=[]
    print (embeddings)
    for sent_i,sent in enumerate(embeddings):
        key=sentences[sent_i]
        if 'play' in key:
            i= key.index('play')

        elif 'bright' in key:
            i= key.index('bright')
        elif 'light' in key:
            i=key.index('light')
        elif 'smart' in key:
            i=key.index('smart')
        vectors.append(np.array(sent[i]))

    print (vectors)
    return np.stack(vectors,axis=0),['\t'.join(sentence) for sentence in sentences]

def elmo_small(hdf5_file):
    f=h5py.File(hdf5_file)
    vectors=[]
    keys=f.keys()
    keywords=[]
    for key in keys:
        print (key)
        words=key.split('\t')
        if 'play' in key:
            i= words.index('play')

        elif 'bright' in key:
            i= words.index('bright')
        elif 'light' in key:
            i=words.index('light')
        elif 'smart' in key:
            i=words.index('smart')
        print (i)
        keywords.append(words[i])
        vectors.append(f[key][i])
    return vectors,keys

if __name__=='__main__':
    vectors,keys=elmo_small('./models/ELMoForManyLangs/english_elmo/test_singleword.ly2.hdf5')
    # vectors,keys=elmo_large('../../evaluation/test.txt')
    # print (vectors.shape)
    vectors=np.array(vectors)
    vectors=vector_heatmap(vectors)
    show_heatmap(vectors,keys)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2,n_init=8).fit(vectors)
    print (kmeans.labels_)




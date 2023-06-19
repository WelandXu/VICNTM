import os
import re
import time
import argparse
import string
import pickle
import json

import numpy as np
from scipy import sparse
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

# Maximum / minimum document frequency
max_df = 0.7
min_df = 100  # choose desired value for min_df

# Read stopwords
with open('stops.txt', 'r') as f:
    stops = f.read().split('\n')

# parse into documents
def is_document_start(line):
    if len(line) < 4:
        return False
    if line[0] == '=' and line[-1] == '=':
        if line[2] != '=':
            return True
        else:
            return False
    else:
        return False


def token_list_per_doc(input_dir, token_file):
    lines_list = []
    line_prev = ''
    prev_line_start_doc = False
    with open(os.path.join(input_dir, token_file), 'r', encoding='utf-8') as f:
        for l in f:
            line = l.strip()
            if prev_line_start_doc and line:
                # the previous line should not have been start of a document!
                lines_list.pop()
                lines_list[-1] = lines_list[-1] + ' ' + line_prev

            if line:
                if is_document_start(line) and not line_prev:
                    lines_list.append(line)
                    prev_line_start_doc = True
                else:
                    lines_list[-1] = lines_list[-1] + ' ' + line
                    prev_line_start_doc = False
            else:
                prev_line_start_doc = False
            line_prev = line

    print("{} documents parsed!".format(len(lines_list)))
    return lines_list

data_dir='/data/wikitext-103'

train_file = 'wiki.train.tokens'
train_doc_list = token_list_per_doc(data_dir, train_file)
val_file = 'wiki.valid.tokens'
test_file = 'wiki.test.tokens'
val_doc_list = token_list_per_doc(data_dir, val_file)
test_doc_list = token_list_per_doc(data_dir, test_file)


init_docs_tr = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', train_doc_list[doc]) for doc in range(len(train_doc_list))]
init_docs_va = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', val_doc_list[doc]) for doc in range(len(val_doc_list))]
init_docs_ts = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', test_doc_list[doc]) for doc in range(len(test_doc_list))]

def contains_punctuation(w):
    return any(char in string.punctuation for char in w)

def contains_numeric(w):
    return any(char.isdigit() for char in w)
    
init_docs = init_docs_tr + init_docs_ts + init_docs_va
init_docs = [[w.lower() for w in init_docs[doc] if not contains_punctuation(w)] for doc in range(len(init_docs))]
init_docs = [[w for w in init_docs[doc] if not contains_numeric(w)] for doc in range(len(init_docs))]
init_docs = [[w for w in init_docs[doc] if len(w)>1] for doc in range(len(init_docs))] 
init_docs = [" ".join(init_docs[doc]) for doc in range(len(init_docs))]

# Create count vectorizer
print('counting document frequency of words...')
cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=None)
cvz = cvectorizer.fit_transform(init_docs).sign()

# Get vocabulary
print('building the vocabulary...')
sum_counts = cvz.sum(axis=0)
v_size = sum_counts.shape[1]
sum_counts_np = np.zeros(v_size, dtype=int)
for v in range(v_size):
    sum_counts_np[v] = sum_counts[0,v]
word2id = dict([(w, cvectorizer.vocabulary_.get(w)) for w in cvectorizer.vocabulary_])
id2word = dict([(cvectorizer.vocabulary_.get(w), w) for w in cvectorizer.vocabulary_])
del cvectorizer
print('  initial vocabulary size: {}'.format(v_size))

# Sort elements in vocabulary
idx_sort = np.argsort(sum_counts_np)
vocab_aux = [id2word[idx_sort[cc]] for cc in range(v_size)]

# Filter out stopwords (if any)
vocab_aux = [w for w in vocab_aux if w not in stops]
print('  vocabulary size after removing stopwords from list: {}'.format(len(vocab_aux)))

# Create dictionary and inverse dictionary
vocab = vocab_aux
del vocab_aux
word2id = dict([(w, j) for j, w in enumerate(vocab)])
id2word = dict([(j, w) for j, w in enumerate(vocab)])

# Split in train/test/valid 70/15/415 
print('tokenizing documents and splitting into train/test/valid...')
trSize = round(len(init_docs) * 0.7)
tsSize = round(len(init_docs) * 0.15)
vaSize = len(init_docs) - trSize - tsSize
np.random.seed(42)
idx_permute = np.random.permutation(trSize+vaSize).astype(int)

# Remove words not in train_data
vocab = list(set([w for idx_d in range(trSize) for w in init_docs[idx_permute[idx_d]].split() if w in word2id]))
word2id = dict([(w, j) for j, w in enumerate(vocab)])
id2word = dict([(j, w) for j, w in enumerate(vocab)])
print('  vocabulary after removing words not in train: {}'.format(len(vocab)))

path_save = '/data/wiki_min_df_' + str(min_df) + '/'
if not os.path.isdir(path_save):
    os.system('mkdir -p ' + path_save)
docs_txt = '\n'.join([' '.join([w for w in init_docs[i].split() if w in word2id]) for i in range(len(init_docs))])
with open(path_save+'corpus_wiki_mindf_100_tok30.txt','w') as f:
    f.write(docs_txt)
del docs_txt
# Split in train/test/valid
docs_all = [[word2id[w] for w in init_docs[i].split() if w in word2id] for i in range(len(init_docs))]
docs_tr = [[word2id[w] for w in init_docs[idx_permute[idx_d]].split() if w in word2id] for idx_d in range(trSize)]
docs_va = [[word2id[w] for w in init_docs[idx_permute[idx_d+trSize]].split() if w in word2id] for idx_d in range(vaSize)]
docs_ts = [[word2id[w] for w in init_docs[idx_d+trSize+vaSize].split() if w in word2id] for idx_d in range(tsSize)]

print('  number of documents (all): {} [this should be equal to {}]'.format(len(docs_all), len(init_docs)))
print('  number of documents (train): {} [this should be equal to {}]'.format(len(docs_tr), trSize))
print('  number of documents (test): {} [this should be equal to {}]'.format(len(docs_ts), tsSize))
print('  number of documents (valid): {} [this should be equal to {}]'.format(len(docs_va), vaSize))

# Remove empty documents
print('removing empty documents...')

def remove_empty(in_docs):
    return [doc for doc in in_docs if doc!=[]]
def remove_short(in_docs):
    return [doc for doc in in_docs if doc!=[]]

docs_all = remove_short(docs_all)
docs_tr = remove_short(docs_tr)
docs_ts = remove_short(docs_ts)
docs_va = remove_short(docs_va)

# Remove test documents with length=1
docs_ts = [doc for doc in docs_ts if len(doc)>1]

# Getting lists of words and doc_indices
print('creating lists of words...')

def create_list_words(in_docs):
    return [x for y in in_docs for x in y]

words_all = create_list_words(docs_all)
words_tr = create_list_words(docs_tr)
words_ts = create_list_words(docs_ts)
words_va = create_list_words(docs_va)

print('  len(words_all): ', len(words_all))
print('  len(words_tr): ', len(words_tr))
print('  len(words_ts): ', len(words_ts))
print('  len(words_va): ', len(words_va))

# Get doc indices
print('getting doc indices...')

def create_doc_indices(in_docs):
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(x) for y in aux for x in y]

doc_indices_all = create_doc_indices(docs_all)
doc_indices_tr = create_doc_indices(docs_tr)
doc_indices_ts = create_doc_indices(docs_ts)
doc_indices_va = create_doc_indices(docs_va)

print('  len(np.unique(doc_indices_all)): {} [this should be {}]'.format(len(np.unique(doc_indices_all)), len(docs_all)))
print('  len(np.unique(doc_indices_tr)): {} [this should be {}]'.format(len(np.unique(doc_indices_tr)), len(docs_tr)))
print('  len(np.unique(doc_indices_ts)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts)), len(docs_ts)))
print('  len(np.unique(doc_indices_va)): {} [this should be {}]'.format(len(np.unique(doc_indices_va)), len(docs_va)))

# Number of documents in each set
n_docs_all = len(docs_all)
n_docs_tr = len(docs_tr)
n_docs_ts = len(docs_ts)
n_docs_va = len(docs_va)

# Remove unused variables
del docs_all
del docs_tr
del docs_ts
del docs_va

# Create bow representation
print('creating bow representation...')

def create_bow(doc_indices, words, n_docs, vocab_size):
    return sparse.coo_matrix(([1]*len(doc_indices),(doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()

bow_all = create_bow(doc_indices_all, words_all, n_docs_all, len(vocab))
bow_tr = create_bow(doc_indices_tr, words_tr, n_docs_tr, len(vocab))
bow_ts = create_bow(doc_indices_ts, words_ts, n_docs_ts, len(vocab))
bow_va = create_bow(doc_indices_va, words_va, n_docs_va, len(vocab))

del words_all
del words_tr
del words_ts
del words_va
del doc_indices_tr
del doc_indices_ts
del doc_indices_va

# Write the vocabulary to a file

with open(path_save + 'vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
with open(path_save + 'vocab.json','w') as f:
    json.dump(vocab, f)
del vocab

# Split bow intro token/value pairs
print('splitting bow intro token/value pairs and saving to disk...')

def split_bow(bow_in, n_docs):
    indices = [[w for w in bow_in[doc,:].indices] for doc in range(n_docs) if len(bow_in[doc,:].indices)>=30]
    counts = [[c for c in bow_in[doc,:].data] for doc in range(n_docs) if len(bow_in[doc,:].indices)>=30]
    print(len(indices),len(counts))
    return indices, counts

def _to_numpy_array(documents):
    return np.array([[np.array(doc) for doc in documents]], dtype=object).squeeze()


bow_all_tokens, bow_all_counts = split_bow(bow_all, n_docs_all)
corpus_dataset={
    'tokens': _to_numpy_array(bow_all_tokens),
    'counts': _to_numpy_array(bow_all_counts)
}
del bow_all
del bow_all_tokens
del bow_all_counts

bow_tr_tokens, bow_tr_counts = split_bow(bow_tr, n_docs_tr)
train_dataset={
    'tokens': _to_numpy_array(bow_tr_tokens),
    'counts': _to_numpy_array(bow_tr_counts)
}
del bow_tr
del bow_tr_tokens
del bow_tr_counts

bow_ts_tokens, bow_ts_counts = split_bow(bow_ts, n_docs_ts)
test_dataset={
    'test':{
        'tokens': _to_numpy_array(bow_ts_tokens),
        'counts': _to_numpy_array(bow_ts_counts)
    }
}
del bow_ts
del bow_ts_tokens
del bow_ts_counts


bow_va_tokens, bow_va_counts = split_bow(bow_va, n_docs_va)
val_dataset={
    'val':{
        'tokens': _to_numpy_array(bow_va_tokens),
        'counts': _to_numpy_array(bow_va_counts)
    }
}
del bow_va
del bow_va_tokens
del bow_va_counts

dataset={
    'all':corpus_dataset,
    'train':train_dataset,
    'validation':val_dataset,
    'test':test_dataset
}
with open(path_save + 'dataset_wiki103.pkl', 'wb') as f:
    pickle.dump(dataset,f)

print('Data ready !!')
print('*************')



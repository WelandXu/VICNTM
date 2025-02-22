from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pickle
import random
from scipy import sparse
import itertools
from scipy.io import savemat, loadmat
import re
import string
import os
import json
from itertools import chain
import sys
import csv
# Maximum / minimum document frequency
max_df = 0.7
min_df = 100  # choose desired value for min_df

# Read stopwords
with open('stops.txt', 'r') as f:
    stops = f.read().split('\n')

# Read data
print('reading data...')
train_data = fetch_20newsgroups(subset='train')
test_data = fetch_20newsgroups(subset='test')
labels_tr=fetch_20newsgroups(subset='train').target
labels_ts=fetch_20newsgroups(subset='test').target

init_docs_tr = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', train_data.data[doc]) for doc in range(len(train_data.data))]
init_docs_ts = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', test_data.data[doc]) for doc in range(len(test_data.data))]


def contains_punctuation(w):
    return any(char in string.punctuation for char in w)

def contains_numeric(w):
    return any(char.isdigit() for char in w)
    
init_docs = init_docs_tr + init_docs_ts
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

# Split in train/test/valid 48/12/40 
print('tokenizing documents and splitting into train/test/valid...')
# num_docs_tr = round(len(init_docs) * 0.85)
trSize = round(len(init_docs) * 0.48)
tsSize = round(len(init_docs) * 0.40)
vaSize = len(init_docs) - trSize - tsSize
np.random.seed(42)
idx_permute = np.random.permutation(trSize+vaSize).astype(int)

# Remove words not in train_data
# vocab = list(set([w for idx_d in range(trSize) for w in init_docs[idx_permute[idx_d]].split() if w in word2id]))
with open('./vocabulary_20ng.txt','r') as f:
    vocab = f.read().split('\n')
    print(vocab[0])
word2id = dict([(w, j) for j, w in enumerate(vocab)])
id2word = dict([(j, w) for j, w in enumerate(vocab)])
print('  vocabulary after removing words not in train: {}'.format(len(vocab)))

path_save = './20NG_min_df_' + str(min_df) + '/'
if not os.path.isdir(path_save):
    os.system('mkdir -p ' + path_save)
docs_txt = '\n'.join([' '.join([w for w in init_docs[i].split() if w in word2id]) for i in range(len(init_docs))])
with open(path_save+'corpus_20ng_mindf_100_tok30.txt','w') as f:
    f.write(docs_txt)
del docs_txt
# Split in train/test/valid
docs_all = [[word2id[w] for w in init_docs[i].split() if w in word2id] for i in range(len(init_docs))]
docs_tr = [[word2id[w] for w in init_docs[idx_permute[idx_d]].split() if w in word2id] for idx_d in range(trSize)]
docs_va = [[word2id[w] for w in init_docs[idx_permute[idx_d+trSize]].split() if w in word2id] for idx_d in range(vaSize)]
docs_ts = [[word2id[w] for w in init_docs[idx_d+trSize+vaSize].split() if w in word2id] for idx_d in range(tsSize)]

docid_tr = idx_permute[:trSize]

labels = list(chain(labels_tr,labels_ts))
labels_tr = [labels[idx_permute[idx_d]] for idx_d in range(trSize)]
labels_va = [labels[idx_permute[idx_d+trSize]] for idx_d in range(vaSize)]
labels_ts = [labels[idx_d+trSize+vaSize] for idx_d in range(tsSize)]

print('  number of documents (all): {} [this should be equal to {}]'.format(len(docs_all), len(init_docs)))
print('  number of documents (train): {} [this should be equal to {}]'.format(len(docs_tr), trSize))
print('  number of documents (test): {} [this should be equal to {}]'.format(len(docs_ts), tsSize))
print('  number of documents (valid): {} [this should be equal to {}]'.format(len(docs_va), vaSize))

print('number of labels (train): {}'.format(len(labels_tr)))
print('number of labels (test): {}'.format(len(labels_ts)))
print('number of labels (valid): {}'.format(len(labels_va)))

labels_map_all=list(zip(docs_all, labels))
labels_map_tr =list(zip(docs_tr, labels_tr))
docid_map_tr = list(zip(docs_tr, docid_tr))
labels_map_ts =list(zip(docs_ts, labels_ts))
labels_map_va =list(zip(docs_va, labels_va))

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


# ## CREATE DATA FOR OCTIS
# print("creating dataset for octis")
# docs_text_tr = [' '.join([id2word[id] for id in doc]) for doc in docs_tr if len(set(doc))>=30]
# docs_text_va = [" ".join([id2word[id] for id in doc]) for doc in docs_va if len(set(doc))>=30]
# docs_text_ts = [" ".join([id2word[id] for id in doc]) for doc in docs_ts if len(set(doc))>=30]
# print('number of training set for octis: {}'.format(len(docs_text_tr)))
# print('number of validation set for octis: {}'.format(len(docs_text_va)))
# print('number of test set for octis: {}'.format(len(docs_text_ts)))

# data = [(text, "train") for text in docs_text_tr] + \
#        [(text, "valid") for text in docs_text_va] + \
#        [(text, "test") for text in docs_text_ts]

# # Define the output file path
# output_file = "20NG_for_octis.tsv"

# # Write the data to a .tsv file
# with open(output_file, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file, delimiter='\t')
#     # Optionally write the header
#     writer.writerow(["text", "label"])
#     # Write the data rows
#     for text, label in data:
#         writer.writerow([text, label])

# print(f"Data successfully written to {output_file}")

# WE should use the former vocab but not create a new one because the order of it changes.

# print('vocab number is {}'.format(len(vocab)))
# print(word2id)

# vocab_words = vocab
# with open('vocabulary_octis_20ng.txt', "w") as f:
#     f.write('\n'.join(vocab_words))

# print(f"vocabulary_octis_20ng.txt successfully created.")

## OCTIS DATA CREATED.


def update_labels_map(docs, labels_map):
    return [labels_map[i] for i in range(len(labels_map)) if labels_map[i][0] in docs]

labels_map_all = update_labels_map(docs_all, labels_map_all)
labels_map_tr = update_labels_map(docs_tr, labels_map_tr)
docid_map_tr = update_labels_map(docs_tr, docid_map_tr)
labels_map_ts = update_labels_map(docs_ts, labels_map_ts)
labels_map_va = update_labels_map(docs_va, labels_map_va)

print('number of labels_map (train): {}'.format(len(labels_map_tr)))
print('number of labels_map (test): {}'.format(len(labels_map_ts)))
print('number of labels_map (valid): {}'.format(len(labels_map_va)))

labels_all = [labels_map_all[i][1] for i in range(len(labels_map_all))]
labels_tr = [labels_map_tr[i][1] for i in range(len(labels_map_tr))]
docid_tr = [docid_map_tr[i][1] for i in range(len(docid_map_tr))]
labels_ts = [labels_map_ts[i][1] for i in range(len(labels_map_ts))]
labels_va = [labels_map_va[i][1] for i in range(len(labels_map_va))]
# # Split test set in 2 halves
# print('splitting test documents in 2 halves...')
# docs_ts_h1 = [[w for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc in docs_ts]
# docs_ts_h2 = [[w for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc in docs_ts]

# Getting lists of words and doc_indices
print('creating lists of words...')

def create_list_words(in_docs):
    return [x for y in in_docs for x in y]

words_all = create_list_words(docs_all)
words_tr = create_list_words(docs_tr)
words_ts = create_list_words(docs_ts)
# words_ts_h1 = create_list_words(docs_ts_h1)
# words_ts_h2 = create_list_words(docs_ts_h2)
words_va = create_list_words(docs_va)

print('  len(words_all): ', len(words_all))
print('  len(words_tr): ', len(words_tr))
print('  len(words_ts): ', len(words_ts))
# print('  len(words_ts_h1): ', len(words_ts_h1))
# print('  len(words_ts_h2): ', len(words_ts_h2))
print('  len(words_va): ', len(words_va))

# Get doc indices
print('getting doc indices...')

def create_doc_indices(in_docs):
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(x) for y in aux for x in y]

doc_indices_all = create_doc_indices(docs_all)
doc_indices_tr = create_doc_indices(docs_tr)
doc_indices_ts = create_doc_indices(docs_ts)
# doc_indices_ts_h1 = create_doc_indices(docs_ts_h1)
# doc_indices_ts_h2 = create_doc_indices(docs_ts_h2)
doc_indices_va = create_doc_indices(docs_va)

print('  len(np.unique(doc_indices_all)): {} [this should be {}]'.format(len(np.unique(doc_indices_all)), len(docs_all)))
print('  len(np.unique(doc_indices_tr)): {} [this should be {}]'.format(len(np.unique(doc_indices_tr)), len(docs_tr)))
print('  len(np.unique(doc_indices_ts)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts)), len(docs_ts)))
# print('  len(np.unique(doc_indices_ts_h1)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts_h1)), len(docs_ts_h1)))
# print('  len(np.unique(doc_indices_ts_h2)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts_h2)), len(docs_ts_h2)))
print('  len(np.unique(doc_indices_va)): {} [this should be {}]'.format(len(np.unique(doc_indices_va)), len(docs_va)))

# Number of documents in each set
n_docs_all = len(docs_all)
n_docs_tr = len(docs_tr)
n_docs_ts = len(docs_ts)
# n_docs_ts_h1 = len(docs_ts_h1)
# n_docs_ts_h2 = len(docs_ts_h2)
n_docs_va = len(docs_va)

# Remove unused variables
del docs_all
del docs_tr
del docs_ts
# del docs_ts_h1
# del docs_ts_h2
del docs_va

# Create bow representation
print('creating bow representation...')

def create_bow(doc_indices, words, n_docs, vocab_size):
    return sparse.coo_matrix(([1]*len(doc_indices),(doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()

bow_all = create_bow(doc_indices_all, words_all, n_docs_all, len(vocab))
bow_tr = create_bow(doc_indices_tr, words_tr, n_docs_tr, len(vocab))
bow_ts = create_bow(doc_indices_ts, words_ts, n_docs_ts, len(vocab))
# bow_ts_h1 = create_bow(doc_indices_ts_h1, words_ts_h1, n_docs_ts_h1, len(vocab))
# bow_ts_h2 = create_bow(doc_indices_ts_h2, words_ts_h2, n_docs_ts_h2, len(vocab))
bow_va = create_bow(doc_indices_va, words_va, n_docs_va, len(vocab))


labels_map_all=list(zip(bow_all, labels_all))
labels_map_tr =list(zip(bow_tr, labels_tr))
docid_map_tr = list(zip(bow_tr, docid_tr))
labels_map_ts =list(zip(bow_ts, labels_ts))
labels_map_va =list(zip(bow_va, labels_va))

labels_map_all=[labels_map_all[i] for i in range(len(labels_all)) if len(labels_map_all[i][0].indices)>=30]
labels_map_tr=[labels_map_tr[i] for i in range(len(labels_tr)) if len(labels_map_tr[i][0].indices)>=30]
docid_tr =[docid_map_tr[i] for i in range(len(docid_tr)) if len(docid_map_tr[i][0].indices)>=30]
labels_map_ts=[labels_map_ts[i] for i in range(len(labels_ts)) if len(labels_map_ts[i][0].indices)>=30]
labels_map_va=[labels_map_va[i] for i in range(len(labels_va)) if len(labels_map_va[i][0].indices)>=30]

print(" numbers of labels: all_{}, tr_{}, ts_{}, va_{}".format(len(labels_map_all),len(labels_map_tr),len(labels_map_ts),len(labels_map_va)))

# with open('labels.pkl','wb') as f:
#     labels={
#         'all':labels_map_all,
#         'tr':labels_map_tr,
#         'ts':labels_map_ts,
#         'va':labels_map_va
#     }
#     pickle.dump(labels, f)
# print('labels saved.')
# with open('20NG_docid_tr.pkl','wb') as f:
#     pickle.dump(docid_tr,f)
# print('doc id saved.')
# print(vocab[0])
# sys.exit()
del words_all
del words_tr
del words_ts
# del words_ts_h1
# del words_ts_h2
del words_va
del doc_indices_tr
del doc_indices_ts
# del doc_indices_ts_h1
# del doc_indices_ts_h2
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
# for i in range(len(bow_all_tokens)):
#     print(i,len(bow_all_tokens[i]))
# savemat(path_save + 'bow_all_tokens.mat', {'tokens': bow_all_tokens}, do_compression=True)
# savemat(path_save + 'bow_all_counts.mat', {'counts': bow_all_counts}, do_compression=True)
corpus_dataset={
    'tokens': _to_numpy_array(bow_all_tokens),
    'counts': _to_numpy_array(bow_all_counts)
}
del bow_all
del bow_all_tokens
del bow_all_counts

bow_tr_tokens, bow_tr_counts = split_bow(bow_tr, n_docs_tr)
# savemat(path_save + 'bow_tr_tokens.mat', {'tokens': bow_tr_tokens}, do_compression=True)
# savemat(path_save + 'bow_tr_counts.mat', {'counts': bow_tr_counts}, do_compression=True)
train_dataset={
    'tokens': _to_numpy_array(bow_tr_tokens),
    'counts': _to_numpy_array(bow_tr_counts)
}
del bow_tr
del bow_tr_tokens
del bow_tr_counts

bow_ts_tokens, bow_ts_counts = split_bow(bow_ts, n_docs_ts)
# savemat(path_save + 'bow_ts_tokens.mat', {'tokens': bow_ts_tokens}, do_compression=True)
# savemat(path_save + 'bow_ts_counts.mat', {'counts': bow_ts_counts}, do_compression=True)
test_dataset={
    'test':{
        'tokens': _to_numpy_array(bow_ts_tokens),
        'counts': _to_numpy_array(bow_ts_counts)
    }
}
del bow_ts
del bow_ts_tokens
del bow_ts_counts

# bow_ts_h1_tokens, bow_ts_h1_counts = split_bow(bow_ts_h1, n_docs_ts_h1)
# savemat(path_save + 'bow_ts_h1_tokens.mat', {'tokens': bow_ts_h1_tokens}, do_compression=True)
# savemat(path_save + 'bow_ts_h1_counts.mat', {'counts': bow_ts_h1_counts}, do_compression=True)
# test_dataset['test1']={
#     'tokens': _to_numpy_array(bow_ts_h1_tokens),
#     'counts': _to_numpy_array(bow_ts_h1_counts)
# }
# del bow_ts_h1
# del bow_ts_h1_tokens
# del bow_ts_h1_counts

# bow_ts_h2_tokens, bow_ts_h2_counts = split_bow(bow_ts_h2, n_docs_ts_h2)
# savemat(path_save + 'bow_ts_h2_tokens.mat', {'tokens': bow_ts_h2_tokens}, do_compression=True)
# savemat(path_save + 'bow_ts_h2_counts.mat', {'counts': bow_ts_h2_counts}, do_compression=True)
# test_dataset['test2']={
#     'tokens': _to_numpy_array(bow_ts_h2_tokens),
#     'counts': _to_numpy_array(bow_ts_h2_counts)
# # }
# del bow_ts_h2
# del bow_ts_h2_tokens
# del bow_ts_h2_counts

bow_va_tokens, bow_va_counts = split_bow(bow_va, n_docs_va)
# savemat(path_save + 'bow_va_tokens.mat', {'tokens': bow_va_tokens}, do_compression=True)
# savemat(path_save + 'bow_va_counts.mat', {'counts': bow_va_counts}, do_compression=True)
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
with open(path_save + 'dataset_20ng.pkl', 'wb') as f:
    pickle.dump(dataset,f)

print('Data ready !!')
print('*************')


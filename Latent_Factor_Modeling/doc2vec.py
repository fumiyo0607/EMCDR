import pandas as pd
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import gensim
from gensim import corpora
from gensim import models
from gensim.models.doc2vec import LabeledSentence
import os
import pickle
import util

class MyDoc2Vec():
    def __init__(self, vector_size, epochs, min_count=3):
        self.vector_size = vector_size
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
        self.user_vec_dir = '/vector/users/doc2vec/doc2vec_vector_size={}_epochs={}'.format(self.vector_size, epochs)

    def _read_corpus(self, corpuses: dict, tokens_only=False):
        for u, corpus in corpuses.items():
            if tokens_only:
                yield corpus
            else:
                yield gensim.models.doc2vec.TaggedDocument(corpus, [u])

    def train_and_save_model(self, corpus_: dict, save_file_name: str, p=0):
        # read corpus
        corpus = list(self._read_corpus(corpus_))
        # build mode
        self.model.build_vocab(corpus)
        # train model
        self.model.train(corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        # save model
        util.ensure_dir('./model/doc2vec')
        self.model.save("./model/doc2vec/doc2vec_vector_size={}-epochs={}_{}.model".format(self.vector_size, self.model.epochs, save_file_name))

    def get_user_vecs(self, user_ids):
        # user_ids: [0 , (user_num - 1)]
        user_vecs = np.zeros([len(user_ids), self.vector_size])
        for u in user_ids:
            user_vecs[int(u)] = self.model.docvecs[u]
        return user_vecs
    
    @staticmethod
    def load_model(save_file_name: str, vector_size=50, epochs=100):
        model = models.Doc2Vec.load("./model/doc2vec/doc2vec_vector_size={}-epochs={}_{}.model".format(vector_size, epochs, save_file_name))
        return model
    
    @staticmethod
    def create_corpus(data_path):
        df = pd.read_csv(data_path)
        users = df['remap_user_id'].values
        items = df['remap_item_id'].values

        corpus  = defaultdict(list)
        for u, i in zip(users, items):
            corpus[str(u)].append(str(i))
        return corpus
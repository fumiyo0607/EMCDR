from Latent_Factor_Modeling.doc2vec import MyDoc2Vec
from Latent_Factor_Modeling.lda import MyLDA
from Latent_Factor_Modeling import bpr
import numpy as np
import util

def doc2vec_source(vector_size=50, epochs=300):
    ## train and save model
    corpus_s  = MyDoc2Vec.create_corpus('./data/data_s_train_p=0.csv')
    doc2vec_s = MyDoc2Vec(vector_size, epochs)
    doc2vec_s.train_and_save_model(corpus_s, 'source_trained')
    ## save trained user vectors
    users       = list(corpus_s.keys())
    users_vec_s = doc2vec_s.get_user_vecs(users)
    util.ensure_dir('./vector/users/doc2vec/')
    np.save('.{}_source_trained'.format(doc2vec_s.user_vec_dir), users_vec_s)

def doc2vec_target(vector_size=50, epochs=300):
    ## train and save model
    corpus_t  = MyDoc2Vec.create_corpus('./data/data_t_train_p=0.csv')
    doc2vec_t = MyDoc2Vec(vector_size, epochs)
    doc2vec_t.train_and_save_model(corpus_t, 'target_trained')
    ## save trained user vectors
    users       = list(corpus_t.keys())
    users_vec_t = doc2vec_t.get_user_vecs(users)
    util.ensure_dir('./vector/users/doc2vec/')
    np.save('.{}_target_trained'.format(doc2vec_t.user_vec_dir), users_vec_t)

def lda_source(topic_num=50, iters=50):
    bow = MyLDA.create_bow('./data/data_s_train_p=0.csv')
    lda = MyLDA(topic_num=topic_num, iteration=iters)
    print('finish initialize')
    lda.train_and_save_model(bow, 'source_trained')
    users_vec = lda.get_user_vec()
    util.ensure_dir('./vector/users/lda/')
    np.save('.{}_source_trained'.format(lda.user_vec_dir), users_vec)
    return

def lda_target(topic_num=50, iters=50):
    bow = MyLDA.create_bow('./data/data_t_train_p=0.csv')
    lda = MyLDA(topic_num=topic_num, iteration=iters)
    lda.train_and_save_model(bow, 'target_trained')
    users_vec = lda.get_user_vec()
    util.ensure_dir('./vector/users/lda/')
    np.save('.{}_target_trained'.format(lda.user_vec_dir), users_vec)
    return


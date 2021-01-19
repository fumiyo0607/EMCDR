import numpy as np
import pandas as pd
import util
from collections import defaultdict
from Latent_Factor_Modeling.doc2vec import MyDoc2Vec
from parameter_setting import Mapping

def _get_test_users(data_t_test_path='./data/data_t_test_p=0.csv'):
    print(data_t_test_path)
    df = pd.read_csv(data_t_test_path)
    df_test = df[df['user_type']=='test']
    del df
    users = df_test['remap_user_id'].values.tolist()
    items = df_test['remap_item_id'].values.tolist()
    users_test_data = defaultdict(list)
    for u, i in zip(users, items):
        users_test_data[u].append(i)
    return users_test_data

def _get_train_users(data_t_train_path='./data/data_t_train_p=0.csv'):
    df = pd.read_csv(data_t_train_path)
    df_train = df[df['user_type']=='train']
    del df
    users = df_train['remap_user_id'].values.tolist()
    items = df_train['remap_item_id'].values.tolist()
    users_train_data = defaultdict(list)
    for u, i in zip(users, items):
        users_train_data[u].append(i)
    return users_train_data

def doc2vec(setting: Mapping, top_n=30, data_t_path='./data/data_s_train_p=0.csv'):
    users_test_data = _get_test_users()
    users_test = list(users_test_data.keys())
    save_path = './vector/users/doc2vec/doc2vec_vector_size={}_epochs={}_mlp_epochs={}_hidden_layer_num={}_target_mapped.npy'.format(
                        setting.latent_facor_model.facter_size, 
                        setting.latent_facor_model.train_num, 
                        setting.mlp.train_num, 
                        setting.mlp.hidden_layer_size)
    Xt = np.load(save_path)
    hit = []
    model_t = MyDoc2Vec.load_model(
                vector_size    = setting.latent_facor_model.facter_size, 
                epochs         = setting.latent_facor_model.train_num, 
                save_file_name = 'target_trained')

    for u, uid in enumerate(users_test):
        user_vec = Xt[u]
        pred_items = model_t.predict(user_vec, top_n)
        pred_items = [ int(i) for i in pred_items ]
        test_items = users_test_data[uid]
        c = 0
        for i in pred_items:
            if i in test_items:
                c += 1
        hit.append(c/top_n)
    return sum(hit) / len(hit)

def doc2vec_train_target(setting: Mapping, top_n=30):
    users_train_data = _get_train_users()
    users_test = list(users_train_data.keys())
    hit = []
    model_t = MyDoc2Vec.load_model(
                vector_size    = setting.latent_facor_model.facter_size, 
                epochs         = setting.latent_facor_model.train_num, 
                save_file_name = 'target_trained')

    for uid in users_test:
        user_vec = model_t.model.docvecs[str(uid)]
        pred_items = model_t.predict(user_vec, top_n)
        pred_items = [ int(i) for i in pred_items ]
        test_items = users_train_data[uid]
        c = 0
        for i in pred_items:
            if i in test_items:
                c += 1
        hit.append(c/top_n)
    return sum(hit) / len(hit)

def lda(setting: Mapping):
    # add code ..
    return

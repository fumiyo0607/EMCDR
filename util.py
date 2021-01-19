# util
import os
import pickle
import numpy as np
from parameter_setting import Mapping

def save_pickle(path, x):
    path = path +'.pickle'
    with open(path, 'wb') as f:
        pickle.dump(x , f)

def load_pickle(path):
    path = path +'.pickle'
    with open(path, 'rb') as f:
        x = pickle.load(f)
    return x

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_np_user_vector(path_s, path_t):
    U_s       = np.load(path_s)
    U_t_train = np.load(path_t)
    U_s_train = U_s[:U_t_train.shape[0], :]
    U_s_test  = U_s[U_t_train.shape[0]:,:]
    return U_s_train, U_s_test, U_t_train 

def trained_latent_factor_vec_parh(setting: Mapping):
    if setting.latent_facor_model.model == 'doc2vec':
        path_s = './vector/users/{}/{}_vector_size={}_epochs={}_source_trained.npy'.format(
            setting.latent_facor_model.model,
            setting.latent_facor_model.model,
            setting.latent_facor_model.facter_size,
            setting.latent_facor_model.train_num,
        )
        path_t = './vector/users/{}/{}_vector_size={}_epochs={}_target_trained.npy'.format(
            setting.latent_facor_model.model,
            setting.latent_facor_model.model,
            setting.latent_facor_model.facter_size,
            setting.latent_facor_model.train_num,
        )
    elif setting.latent_facor_model.model == 'lda':
        path_s = './vector/users/{}/{}_topic_num={}_iter={}_source_trained.npy'.format(
            setting.latent_facor_model.model,
            setting.latent_facor_model.model,
            setting.latent_facor_model.facter_size,
            setting.latent_facor_model.train_num,
        )
        path_t = './vector/users/{}/{}_topic_num={}_iter={}_target_trained.npy'.format(
            setting.latent_facor_model.model,
            setting.latent_facor_model.model,
            setting.latent_facor_model.facter_size,
            setting.latent_facor_model.train_num,
        )
    return path_s, path_t
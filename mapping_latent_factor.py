from Latent_Space_Mapping.mlp import MLP
import util
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tqdm.notebook import tqdm

'''**** path example ****
    path_s = './vector/users/doc2vec/doc2vec_vector_size=50_epochs=300_source_trained.npy'
    path_t = './vector/users/doc2vec/doc2vec_vector_size=50_epochs=300_target_trained.npy'
'''

def mapping(path_s, path_t, training_epochs=300, learning_rate=0.03, beta=0.001, display_step=10):
    ## load trained users vector
    U_s_train, U_s_test, U_t_train = util.load_np_user_vector(path_s, path_t)
    print('source domain train : ' , U_s_train.shape)
    print('source domain test  : ' , U_s_test.shape)
    print('target domain       : ' , U_t_train.shape)

    ## clear graph
    tf.reset_default_graph()

    ## train and mapping 
    model = MLP(U_s_train.T, U_t_train.T, beta, learning_rate)
    Xt  = []
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        loss = model.train(sess, U_s_train.T, U_t_train.T, training_epochs, display_step)
        for u in tqdm(range(U_s_test.shape[0])):
            xs = U_s_test.T[:,u]
            xs = xs.reshape([model.k,1])
            xt = model.latent_space_mapping(sess, xs)
            Xt.append(xt.reshape([model.k]).tolist())
    Xt = np.array(Xt)
    path_save = path_t.replace('trained.npy', 'mapped')
    np.save(path_save, Xt)
    return loss, Xt
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from tensorflow.contrib import layers
import util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class BPR():
    def __init__(self, data_file_path, k, beta):
        _ , items , user_ratings = self._load_data(data_file_path)
        self.m = len(user_ratings)
        self.n = len(items)
        self.k = k
        
        # 1. set initial parameters
        self.u = tf.placeholder(tf.int32, [None])
        self.i = tf.placeholder(tf.int32, [None])
        self.j = tf.placeholder(tf.int32, [None])

        self.U = tf.get_variable("U", [self.m, self.k], initializer=tf.random_normal_initializer(0, 0.1))
        self.V = tf.get_variable("V", [self.n, self.k], initializer=tf.random_normal_initializer(0, 0.1))
        
        u_emb = tf.nn.embedding_lookup(self.U, self.u)
        i_emb = tf.nn.embedding_lookup(self.V, self.i)
        j_emb = tf.nn.embedding_lookup(self.V, self.j)

        # 3. build model
        self.pred = tf.reduce_mean(tf.multiply(u_emb, (i_emb - j_emb)), 1, keepdims = True)
        self.auc = tf.reduce_mean(tf.to_float(self.pred > 0))
        self.regu = layers.l2_regularizer(beta)(u_emb)
        self.regi = layers.l2_regularizer(beta)(i_emb)
        self.regj = layers.l2_regularizer(beta)(j_emb)

        self.cost = self.regu + self.regi + self.regj - tf.reduce_mean(tf.log(tf.sigmoid(self.pred)))
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        print('parameter setting done..')
    
    def _load_data(self, file_path):
        data = pd.read_csv(file_path)
        users = list(data['remap_user_id'].values)
        items = list(data['remap_item_id'].values)
        
        user_ratings = defaultdict(list)
        for u, i in zip(users, items):
            user_ratings[u].append(i)
        
        users = [ u for u in range(len(set(users)))]
        items = [ i for i in range(len(set(items)))]
        return users, items, user_ratings
    
    # test は必要ない
    def _generate_test(self, user_ratings):
        user_ratings_test = {}
        for user in user_ratings:
            user_ratings_test[user] = random.sample(user_ratings[user], 1)[0]
        return user_ratings_test
    
    def _generate_train_batch(self, user_ratings, user_ratings_test, n, batch_size=124):
        b_max = int(len(user_ratings.keys()) / batch_size) + 1
        users = list(user_ratings.keys())
        for b in range(1, b_max):
            t = []
            idx_start = batch_size * b
            idx_end = batch_size * (b+1)
            if (b+1) == b_max:
                idx_end = len(users)
            users_b = users[idx_start:idx_end]
            for u in users_b:
                i_u = list(user_ratings[u])
                i_u.remove(user_ratings_test[u])
                for i in i_u:
                    j = random.randint(0, n-1)
                    while j in user_ratings[u]:
                        j = random.randint(0, n-1)
                    t.append([u, i, j])
            train_batch = np.asarray(t)
            yield train_batch
        
    def train(self, sess, data_file_path, save_file_name, training_epochs, display_step=10):
        _ , _ , user_ratings = self._load_data(data_file_path)
        user_ratings_test    = self._generate_test(user_ratings)

        for epoch in range(training_epochs):
            avg_cost = 0
            avg_auc  = 0
            p        = 0

            if (epoch + 1) % display_step == 0:
                # display step ..
                for uij in self._generate_train_batch(user_ratings, user_ratings_test, self.n):
                    batch_cost, batch_auc = sess.run([self.cost, self.auc], feed_dict={self.u: uij[:, 0], self.i: uij[:, 1], self.j: uij[:, 2]})
                    avg_cost += batch_cost
                    avg_auc  += batch_auc
                    p        += 1
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost / p), "auc", "{:.9f}".format(avg_auc / p))
            else:
                # train step ..
                for uij in self._generate_train_batch(user_ratings, user_ratings_test, self.n):
                    _ = sess.run([self.train_step], feed_dict={self.u: uij[:, 0], self.i: uij[:, 1], self.j: uij[:, 2]})

        # print model parameters
        variable_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable:", k)
            print("Shape: ", v.shape)
            print(v)
        
        # save mdoel
        saver = tf.train.Saver()
        util.ensure_dir('./model/bpr/')
        saver.save(sess, "./model/bpr/{}_k={}".format(save_file_name, self.k))
        print("Optimization Finished!")


################ MAIN ################

if __name__ == "__main__":
    k = 20 
    beta = 0.0001 
    learning_rate = 0.01
    training_epochs = 100
    display_step = 10
    data_file_path = '../data/data_s_train_p=0.csv'
    
    bpr = BPR(data_file_path, k, beta)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        bpr.train(sess, data_file_path, 'bpr_s', training_epochs)
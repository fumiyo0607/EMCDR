import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from collections import defaultdict
import util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

class MLP():
    def __init__(self, input_Vs, input_Vt, beta, learning_rate, hidden_layer_size=None, activation_fuc=None):
            self.k, self.m     = np.shape(input_Vs)
            self.beta          = beta
            self.learning_rate = learning_rate
            ######## NOTATION #######
            # k : embedding dimension
            # m : item / user num 
            #########################
            if hidden_layer_size == None:
                self.hidden_layer_size = int(2 * self.k)
            else:
                self.hidden_layer_size = hidden_layer_size

            # 1. set initial value
            self.w1 = tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.k], stddev = 0.1), name="w1")
            self.b1 = tf.Variable(tf.zeros([self.hidden_layer_size, 1]), name="b1")

            self.w2 = tf.Variable(tf.zeros([self.k, self.hidden_layer_size]), name="w2")
            self.b2 = tf.Variable(tf.zeros([self.k, 1]), name="b2")

            self.Vs = tf.placeholder(tf.float32,[self.k, self.m])
            self.Vt = tf.placeholder(tf.float32,[self.k, self.m])
            self.xs = tf.placeholder(tf.float32,[self.k, 1])

            # 2. build model
            self.hidden1   = tf.nn.tanh(tf.matmul(self.w1, self.Vs)+self.b1)
            self.hidden1_x = tf.nn.tanh(tf.matmul(self.w1, self.xs)+self.b1)

            self.reg_w1 = layers.l2_regularizer(beta)(self.w1)
            self.reg_w2 = layers.l2_regularizer(beta)(self.w2)

            if activation_fuc == None:
                self.pred   = tf.matmul(self.w2, self.hidden1) + self.b2
                self.pred_x = tf.matmul(self.w2, self.hidden1_x) + self.b2
            elif activation_fuc == 'sigmoid':
                self.pred   = tf.nn.sigmoid(tf.matmul(self.w2, self.hidden1) + self.b2)
                self.pred_x = tf.nn.sigmoid(tf.matmul(self.w2, self.hidden1_x) + self.b2)
            elif activation_fuc == 'relu':
                self.pred   = tf.nn.relu(tf.matmul(self.w2, self.hidden1) + self.b2)
                self.pred_x = tf.nn.relu(tf.matmul(self.w2, self.hidden1_x) + self.b2)
            
            self.cost = tf.reduce_mean(tf.square(self.Vt - self.pred)) + self.reg_w1 + self.reg_w2
            self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)
    
    def train(self, sess, input_Vs, input_Vt, training_epochs=1000, display_step=100):
        avg_costs = defaultdict(float)

        for epoch in range(training_epochs):
            sess.run(self.train_step, feed_dict={self.Vs: input_Vs, self.Vt: input_Vt})

            if (epoch + 1) % display_step == 0:
                avg_cost = sess.run(self.cost, feed_dict={self.Vs: input_Vs, self.Vt: input_Vt})
                avg_costs[epoch+1] = avg_cost
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        # print variable
        variable_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable:", k)
            print("Shape: ", v.shape)
        
        # save model
        saver = tf.train.Saver()
        util.ensure_dir('./model/mlp/')
        saver.save(sess, "./model/mlp/mlp_beta={}_learning_rate={}_epochs={}".format(self.beta, self.learning_rate, training_epochs))
        print("Optimization Finished!")
        
        return avg_costs
    
    def latent_space_mapping(self, sess, x_s):
        return sess.run(self.pred_x, feed_dict={self.xs: x_s})
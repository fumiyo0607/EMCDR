import util
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation

class MyLDA():
    def __init__(self, topic_num, iteration):
        self.topic_num = topic_num
        self.iteration = iteration
        self.model = LatentDirichletAllocation(
                        n_components    = topic_num,  
                        max_iter        = iteration,
                        learning_method = 'batch', 
                        random_state    = 0, 
                        n_jobs          = -1
                    )
        self.bow = None
        self.user_vec_dir = '/vector/users/lda/lda_topic_num={}_iter={}'.format(self.topic_num, self.iteration)
    
    def train_and_save_model(self, bow, save_file_name: str):
        self.model.fit(bow)
        self.bow = bow
        print('finish training')
        util.ensure_dir('./model/lda')
        util.save_pickle('./model/lda/lda_topic_num={}-iter={}_{}'.format(self.topic_num, self.iteration, save_file_name), self)
        return
     
    def get_user_vec(self, user_ids=None):
        if self.bow == None:
            print('prease train model..')
            return
        else:
            if user_ids == None:
                user_vec = self.model.transform(self.bow)
            else:
                user_vec = self.model.transform(self.bow[user_ids,:])
            return user_vec
    
    def predict(self, user_vec, top_n):
        beta = self.model.components_ / self.model.components_.sum(axis=0)[np.newaxis,:]
        beta = beta.T # [item_num, topic_num]
        item_num = beta.shape[0]

        prob = defaultdict(dict)
        for i in range(item_num):
            beta_i = beta[i]
            prob[i] = 0
            for t in range(self.topic_num):
                prob[i] += user_vec[t] * beta_i[t]
        rec_items = sorted(prob.items(), key=lambda x:x[1], reverse=True)
        rec_items = [i for i, p in rec_items]
        rec_items_top_n = rec_items[:top_n]

        del prob, rec_items
        return rec_items_top_n
    
    @staticmethod
    def create_bow(data_path):
        df = pd.read_csv(data_path)
        users = df['remap_user_id'].values.tolist()
        items = df['remap_item_id'].values.tolist()
        del df
        user_num = max(users) + 1
        item_num = max(items) + 1
        print('user num : ', user_num)
        print('item num : ', item_num)

        data = defaultdict(dict)
        for u, i in zip(users, items):
            data[u][i] = 0
        for u, i in zip(users, items):
            data[u][i] += 1
        
        row_users   = []
        col_items   = []
        matrix_data = []
        for u, items in data.items():
            for i in items:
                row_users.append(u)
                col_items.append(i)
                matrix_data.append(data[u][i])

        bow = csr_matrix((matrix_data, (row_users, col_items)))
        return bow
            
    @staticmethod
    def load_model(save_file_name: str, topic_num=50, iteration=10):
        # MyLDA
        model = util.load_pickle("./model/lda/lda_topic_num={}-iter={}_{}".format(topic_num, iteration, save_file_name))
        return model
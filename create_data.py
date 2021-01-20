import pandas as pd
from collections import defaultdict
import numpy as np
import random
import itertools
from tqdm.notebook import tqdm

def load(min_num_i = 20, min_num_u = 20, source_data='Books.csv', target_data='Movies_and_TV.csv'):
    df_s = pd.read_csv('./data/{}'.format(source_data), header=None)
    df_t = pd.read_csv('./data/{}'.format(target_data), header=None)

    df_s.rename(columns={0: 'item_id', 1: 'user_id', 2: 'rating', 3:'timestamp'}, inplace=True)
    df_s.head()

    df_t.rename(columns={0: 'item_id', 1: 'user_id', 2: 'rating', 3:'timestamp'}, inplace=True)
    df_t.head()

    """Book"""
    user_s_c = defaultdict(int)

    users = df_s['user_id'].values
    for u in users:
        user_s_c[u] += 1

    user_s_cc = list(user_s_c.values())
    print('source doain user count', source_data)
    print('max:', max(user_s_cc))
    print('min:', min(user_s_cc))

    """Movie and TV"""
    user_t_c = defaultdict(int)

    users = df_t['user_id'].values
    for u in users:
        user_t_c[u] += 1

    user_t_cc = list(user_t_c.values())
    print('target doain user count', target_data)
    print('max:', max(user_t_cc))
    print('min:', min(user_t_cc))

    """item の抽出"""

    # source domain
    item_s_c = defaultdict(int)

    items = df_s['item_id'].values
    for i in items:
        item_s_c[i] += 1

    item_s_val = []

    for i, c in item_s_c.items():
        if c >= min_num_i:
            item_s_val.append(i)

    #  target domain
    item_t_c = defaultdict(int)

    items = df_t['item_id'].values
    for i in items:
        item_t_c[i] += 1

    item_t_val = []

    for i, c in item_t_c.items():
        if c >= min_num_i:
            item_t_val.append(i)
    
    print('firtering items done .. !!')
    print('item_s_val_num : ', len(item_s_val))
    print('item_t_val_num : ', len(item_t_val))

    df_s = df_s[df_s['item_id'].isin(item_s_val)]
    df_t = df_t[df_t['item_id'].isin(item_t_val)]

    """user の抽出"""

    # source domain
    user_s_c = defaultdict(int)
    users = df_s['user_id'].values
    for u in users:
        user_s_c[u] += 1

    user_s_val = []
    for u, c in user_s_c.items():
        if c >= min_num_u:
            user_s_val.append(u)

    # target domain
    user_t_c = defaultdict(int)
    users = df_t['user_id'].values
    for u in users:
        user_t_c[u] += 1

    user_t_val = []
    for u, c in user_t_c.items():
        if c >= min_num_u:
            user_t_val.append(u)

    user_all_val = []
    for u in tqdm(user_t_val):
        if u in user_s_val:
            user_all_val.append(u)
    
    print('firtering users done .. !!')
    print('user_val_num : ', len(user_all_val))

    df_s = df_s.query('user_id in {}'.format(user_all_val))
    df_t = df_t.query('user_id in {}'.format(user_all_val))

    df_s.to_csv('./data/data_val_s.csv')
    df_t.to_csv('./data/data_val_t.csv')

    """## 02：train / test 用のデータセットを作成"""

    cold_users_item_num = 0

    df_s = pd.read_csv('./data/data_val_s.csv')
    df_t = pd.read_csv('./data/data_val_t.csv')

    # select test users
    users = df_s['user_id'].to_list()
    users = list(set(users))

    random.seed(2021)
    users_test = random.sample(users, int(len(users)*0.1))

    users_train = []
    for u in users:
        if u not in users_test:
            users_train.append(u)

    print('all users num : ', len(users))
    print('train users num : ', len(users_train))
    print('test users num : ', len(users_test))

    # target domain
    # split target domain items to train / test

    data_t_all = defaultdict(dict)

    users_t = df_t['user_id'].values
    items_t = df_t['item_id'].values
    times_t = df_t['timestamp'].values

    for u , i, t in zip(tqdm(users_t), items_t, times_t):
        data_t_all[u][i] = t

    data_t_test = defaultdict(list)
    data_t_train = defaultdict(list)

    for u, item_datas in tqdm(data_t_all.items()):
        item_sorted = sorted(item_datas.items(), key=lambda x:x[1])
        item_sorted = [ i for i, t in item_sorted]
        if u in users_test:
            data_t_test[u] = item_sorted[cold_users_item_num:]
            data_t_train[u] = item_sorted[:cold_users_item_num]
        else:
            data_t_train[u] = item_sorted

    # source domain
    data_s_all = defaultdict(list)
    users_s = df_s['user_id'].values
    items_s = df_s['item_id'].values

    data_s_test = defaultdict(list)
    data_s_train = defaultdict(list)

    for u, i in zip(users_s, items_s):
        data_s_all[u].append(i)
        
    for u, items in data_s_all.items():
        if u in users_train:
            data_s_train[u] = items
        elif u in users_test:
            data_s_test[u] = items  

    print('train users num : ', len(data_s_train.keys()))
    print('test users num : ', len(data_s_test.keys()))

    # 学習データ（data_t_train）に出現しないアイテムを テストデータ（data_t_test）から削除
    data_t_test_ = data_t_test
    data_t_test = defaultdict(list)

    items_appered = list(data_t_train.values())
    items_appered = list(itertools.chain.from_iterable(items_appered))
    items_appered = list(set(items_appered))

    for u, item_datas in tqdm(data_t_test_.items()):
        for i in item_datas:
            if i in items_appered:
                data_t_test[u].append(i)

    """## 03：user / item を remap

    User
    """

    # source / target domain: train users
    user_s_train = list(data_s_train.keys())
    user_s_train = list(set(user_s_train))

    remap_user_train = defaultdict(int)
    for idx, u in enumerate(user_s_train):
        remap_user_train[u] = idx

    # source / target domain: test users
    user_t_test = list(data_t_test.keys())
    user_t_test = list(set(user_t_test))

    remap_user_test = defaultdict(int)
    for idx, u in enumerate(user_t_test):
        remap_user_test[u] = idx

    # source / target domain: train users
    remap_user = defaultdict(int)
    for idx, u in enumerate(users_train):
        remap_user[u] = idx

    for idx, u in enumerate(users_test):
        remap_user[u] = idx + len(users_train)

    """Item"""

    # souce domain: train items
    item_s_train = list(data_s_train.values())

    item_s_train = list(itertools.chain.from_iterable(item_s_train))
    item_s_train = list(set(item_s_train))

    remap_item_s_train = defaultdict(int)
    for idx, i in enumerate(item_s_train):
        remap_item_s_train[i] = idx

    # target domain: train items
    item_t_train = list(data_t_train.values())

    item_t_train = list(itertools.chain.from_iterable(item_t_train))
    item_t_train = list(set(item_t_train))

    remap_item_t_train = defaultdict(int)
    for idx, i in enumerate(item_t_train):
        remap_item_t_train[i] = idx

    # target domain: test items
    item_t_test = list(data_t_test.values())

    item_t_test = list(itertools.chain.from_iterable(item_t_test))
    item_t_test = list(set(item_t_test))

    remap_item_t_test = defaultdict(int)
    for idx, i in enumerate(item_t_test):
        remap_item_t_test[i] = idx

    """## 04：データを出力"""

    users_s = []
    items_s = []

    for u, item_datas in data_s_all.items():
        for i in item_datas:
            users_s.append(u)
            items_s.append(i)

    df_s_train = pd.DataFrame(data={'user_id':users_s, 'item_id': items_s})
    df_s_train.head(3)

    def remap_user_id(u_id):
        return remap_user[u_id]

    def remap_item_id_s_train(i_id):
        return remap_item_s_train[i_id]

    def user_type(u_id):
        if u_id in users_test:
            return 'test'
        else:
            return 'train'

    df_s_train['remap_user_id'] = df_s_train['user_id'].map(remap_user_id)
    df_s_train['remap_item_id'] = df_s_train['item_id'].map(remap_item_id_s_train)
    df_s_train['user_type'] = df_s_train['user_id'].map(user_type)
    df_s_train.sort_values('remap_user_id', inplace=True)

    print(max(df_s_train['remap_user_id'].values))
    print(min(df_s_train['remap_user_id'].values))

    print(max(df_s_train['remap_item_id'].values))
    print(min(df_s_train['remap_item_id'].values))

    df_s_train.to_csv('./data/data_s_train_p={}.csv'.format(cold_users_item_num))

    users_t_train = []
    items_t_train = []

    for u, item_datas in data_t_train.items():
        for i in item_datas:
            users_t_train.append(u)
            items_t_train.append(i)

    df_t_train = pd.DataFrame(data={'user_id':users_t_train, 'item_id': items_t_train})
    df_t_train.head(3)

    def remap_item_id_t_train(i_id):
        return remap_item_t_train[i_id]

    df_t_train['remap_user_id'] = df_t_train['user_id'].map(remap_user_id)
    df_t_train['remap_item_id'] = df_t_train['item_id'].map(remap_item_id_t_train)
    df_t_train['user_type'] = df_t_train['user_id'].map(user_type)
    df_t_train.sort_values('remap_user_id', inplace=True)

    print(max(df_t_train['remap_user_id'].values))
    print(min(df_t_train['remap_user_id'].values))

    print(max(df_t_train['remap_item_id'].values))
    print(min(df_t_train['remap_item_id'].values))

    df_t_train.to_csv('./data/data_t_train_p={}.csv'.format(cold_users_item_num))

    users_t_test = []
    items_t_test = []

    for u, item_datas in data_t_test.items():
        for i in item_datas:
            users_t_test.append(u)
            items_t_test.append(i)

    df_t_test = pd.DataFrame(data={'user_id':users_t_test, 'item_id': items_t_test})
    df_t_test.head(3)

    def remap_item_id_t_test(i_id):
        return remap_item_t_test[i_id]

    df_t_test['remap_user_id'] = df_t_test['user_id'].map(remap_user_id)
    df_t_test['remap_item_id'] = df_t_test['item_id'].map(remap_item_id_t_test)
    df_t_test['user_type'] = df_t_test['user_id'].map(user_type)
    df_t_test.sort_values('remap_user_id', inplace=True)

    print(max(df_t_test['remap_user_id'].values))
    print(min(df_t_test['remap_user_id'].values))

    print(max(df_t_test['remap_item_id'].values))
    print(min(df_t_test['remap_item_id'].values))

    df_t_test.to_csv('./data/data_t_test_p={}.csv'.format(cold_users_item_num))
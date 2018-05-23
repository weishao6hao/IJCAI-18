# -*- coding: utf-8 -*-
# from BayesSmooth import smooth_ItemID, smooth_ItembrandID
# from ItemIDBayesSmooth import concatItemDayCVR
# from tools import zuhe, user_shop, user_item, shop_item, user, item
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.externals import joblib
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import time

class solution():
    def __init__(self, evalOrTest='eval'):
        self.n_fold = 5
        self.random_seed = 1
        self.XGBbestNumRounds = 811
        self.LGBMbestNumRounds = 2263
        self.treeNum = 3000
        self.evalOrTest = evalOrTest
        self.nTreeLimit = 30  # only use nTreeLimit trees to generate GBDT feature
        pass

    def readData(self):
        train_ad_data = pd.read_table('./datasets/round1_ijcai_18_train_20180301.txt', sep=' ')
        train_ad_data.drop_duplicates(inplace=True)

        # load the size of trainData
        self.trainLen = len(train_ad_data)
        self.trainlabel = train_ad_data['is_trade']

        test_ad_data = pd.read_table('./datasets/round1_ijcai_18_test_a_20180301.txt', sep=' ')
        self.testInstanceID = test_ad_data['instance_id']

        ad_data = self.appendData(train_ad_data, test_ad_data)

        return ad_data

    def map_hour(self, x):
        if (x >= 1) & (x <= 6):
            return 1
        elif ((x >= 7) & (x <= 9)) or (x == 0):
            return 2
        elif ((x >= 10) & (x <= 18)) or (x == 23):
            return 3
        elif (x >= 19) & (x <= 22):
            return 4
            # else:
            #   return 5

    def deliver(self, x):
        # x=round(x,6)
        jiange = 0.1
        for i in range(1, 20):
            if (x >= 4.1 + jiange * (i - 1)) & (x <= 4.1 + jiange * i):
                return i + 1
            if x == -5:
                return 1

    def deliver1(self, x):
        if (x >= 2) & (x <= 4):
            return 1
        elif (x >= 5) & (x <= 7):
            return 2
        else:
            return 3

    def review(self, x):
        # x=round(x,6)
        jiange = 0.02
        for i in range(1, 30):
            if (x >= 0.714 + jiange * (i - 1)) & (x <= 0.714 + jiange * i):
                return i + 1
            if x == -1:
                return 1

    def service(self, x):
        # x=round(x,6)
        jiange = 0.1
        for i in range(1, 20):
            if (x >= 3.93 + jiange * (i - 1)) & (x <= 3.93 + jiange * i):
                return i + 1
            if x == -1:
                return 1

    def extra_feature(self,data):
        '''
        Some extra feature also have great performance, such as, Whether item is the most expensive/cheapest in this category in the store
        '''
        data['sales_div_pv'] = data.item_sales_level / (1 + data.item_pv_level)
        data['sales_div_pv'] = data.sales_div_pv.map(lambda x: int(10 * x), na_action='ignore')
        number_click_day = data.groupby(['day']).size().reset_index().rename(columns={0:'number_click_day'})
        data = pd.merge(data,number_click_day,'left',on=['day'])
        number_click_hour = data.groupby(['hour']).size().reset_index().rename(columns={0:'number_click_hour'})
        data = pd.merge(data,number_click_hour,'left',on=['hour'])
        temp = data.groupby('item_id')['user_age_level'].nunique().reset_index().rename(columns={'user_age_level': 'number_' + 'user_age_level' + '_query_item'})
        data = pd.merge(data, temp, 'left', on=['item_id'])
        number_category_item = data.groupby(['item_category_list_2','item_id']).size().reset_index().rename(columns={0:'number_category_item'})
        data = pd.merge(data,number_category_item,'left',on=['item_category_list_2','item_id'])
        number_category2 = data.groupby(['item_category_list_2']).size().reset_index().rename(columns={0:'number_category2'})
        data = pd.merge(data,number_category2,'left',on=['item_category_list_2'])
        data['prob_item_id_category2'] = data['number_category_item']/data['number_category2']
        data = data.drop(['number_category2','number_category_item'],axis=1)
        ave_price_category_item = data.groupby(['item_category_list_2','item_id']).mean()['item_price_level'].reset_index().rename(columns={'item_price_level':'ave_price_category_item'})
        data = pd.merge(data,ave_price_category_item,'left',on=['item_category_list_2','item_id'])
        ave_price_category = data.groupby(['item_category_list_2']).mean()['item_price_level'].reset_index().rename(columns={'item_price_level':'ave_price_category'})
        data = pd.merge(data,ave_price_category,'left',on=['item_category_list_2'])
        data['prob_item_price_to_ave_category2'] = data['item_price_level']/data['ave_price_category']

        ave_sales_price_category_item = data.groupby(['item_category_list_2','item_id','item_price_level']).mean()['item_sales_level'].reset_index().rename(columns={'item_sales_level':'ave_sales_price_category_item'})
        data = pd.merge(data,ave_sales_price_category_item,'left',on=['item_category_list_2','item_id','item_price_level'])
        ave_sales_level_category = data.groupby(['item_category_list_2']).mean()['item_sales_level'].reset_index().rename(columns={'item_sales_level':'ave_sales_level_category'})
        data = pd.merge(data,ave_sales_level_category,'left',on=['item_category_list_2'])
        data['prob_ave_category_sales_item_sales'] = data['item_sales_level']/data['ave_sales_level_category']

        max_price_category = data.groupby(['item_category_list_2'])['item_price_level'].max().reset_index().rename(columns={'item_price_level':'max_price_category'})
        data = pd.merge(data,max_price_category,'left',on=['item_category_list_2'])
        data['is_max_price_category'] = data['item_price_level']/data['max_price_category']
        data['is_max_price_category'] = data['is_max_price_category'].map(lambda x: int(x), na_action='ignore')

        min_price_category = data.groupby(['item_category_list_2'])['item_price_level'].min().reset_index().rename(columns={'item_price_level':'min_price_category'})
        data = pd.merge(data,min_price_category,'left',on=['item_category_list_2'])
        data['is_min_price_category'] = data['min_price_category']/data['item_price_level']
        data['is_min_price_category'] = data['is_min_price_category'].map(lambda x: int(x), na_action='ignore')
        data = data.drop(['max_price_category','min_price_category'],axis=1)

        max_sales_category = data.groupby(['item_category_list_2'])['item_sales_level'].max().reset_index().rename(columns={'item_sales_level':'max_sales_category'})
        data = pd.merge(data,max_sales_category,'left',on=['item_category_list_2'])
        data['is_max_sales_category'] = data['item_sales_level']/data['max_sales_category']
        data['is_max_sales_category'] = data['is_max_sales_category'].map(lambda x: int(x), na_action='ignore')

        min_sales_category = data.groupby(['item_category_list_2'])['item_sales_level'].min().reset_index().rename(columns={'item_sales_level':'min_sales_category'})
        data = pd.merge(data,min_sales_category,'left',on=['item_category_list_2'])
        data['is_min_sales_category'] = data['min_sales_category']/data['item_sales_level']
        data['is_min_sales_category'] = data['is_min_sales_category'].map(lambda x: int(x), na_action='ignore')
        data = data.drop(['max_sales_category', 'min_sales_category'], axis=1)

        max_cnt_user_id = data.groupby(['user_id'])['rn_x'].max().reset_index().rename(columns={'rn_x':'max_cnt_user_id'})
        data = pd.merge(data,max_cnt_user_id,'left',on=['user_id'])
        data['is_max_cnt_user_id'] = data['rn_x']/data['max_cnt_user_id']
        data['is_max_cnt_user_id'] = data['is_max_cnt_user_id'].map(lambda x: int(x), na_action='ignore')

        min_cnt_user_id = data.groupby(['user_id'])['rn_x'].min().reset_index().rename(columns={'rn_x':'min_cnt_user_id'})
        data = pd.merge(data,min_cnt_user_id,'left',on=['user_id'])
        data['is_min_cnt_user_id'] = data['min_cnt_user_id']/data['rn_x']
        data['is_min_cnt_user_id'] = data['is_min_cnt_user_id'].map(lambda x: int(x), na_action='ignore')
        data = data.drop(['max_cnt_user_id', 'min_cnt_user_id'], axis=1)
        data['sales_minus_collected'] = data['item_sales_level'] - data['item_collected_level']
        return data

    def describe(self, x):
        # x=round(x,6)
        jiange = 0.1
        for i in range(1, 30):
            if (x >= 3.93 + jiange * (i - 1)) & (x <= 3.93 + jiange * i):
                return i + 1
            if x == -1:
                return 1

    def describe1(self, x):
        if (x >= 2) & (x <= 8):
            return 1
        elif (x >= 9) & (x <= 10):
            return 2
        else:
            return 3

    def zuhe_feature(self, data):
        number_item_user_id = data.groupby(['item_id', 'user_id']).size().reset_index().rename(
            columns={0: 'number_item_user_id'})
        data = pd.merge(data, number_item_user_id, 'left', on=['item_id', 'user_id'])
        data['item_id_user_prob'] = data['number_item_user_id'] / data['number_user_id']
        data['sale_price'] = data['item_sales_level'] + data['item_price_level']
        data['gender_star'] = data['gender0'] + data['user_star_level']

        item_brand_id_user_cnt = data.groupby(['item_brand_id', 'user_id']).size().reset_index().rename(
            columns={0: 'item_brand_id_user_cnt'})
        data = pd.merge(data, item_brand_id_user_cnt, 'left', on=['item_brand_id', 'user_id'])
        data['item_brand_id_user_prob'] = data['item_brand_id_user_cnt'] / data['number_user_id']

        item_id_shop_cnt = data.groupby(['item_id', 'shop_id']).size().reset_index().rename(
            columns={0: 'item_id_shop_cnt'})
        data = pd.merge(data, item_id_shop_cnt, 'left', on=['item_id', 'shop_id'])
        data['item_id_shop_prob'] = data['item_id_shop_cnt'] / data['number_shop_id']

        shop_rev_cnt = data.groupby(['shop_review_positive_rate']).size().reset_index().rename(
            columns={0: 'shop_rev_cnt'})
        data = pd.merge(data, shop_rev_cnt, 'left', on=['shop_review_positive_rate'])
        item_price_level_shop_rev_cnt = data.groupby(
            ['shop_review_positive_rate', 'item_price_level']).size().reset_index().rename(
            columns={0: 'item_price_level_shop_rev_cnt'})
        data = pd.merge(data, item_price_level_shop_rev_cnt, 'left',
                        on=['shop_review_positive_rate', 'item_price_level'])
        data['item_price_level_shop_rev_prob'] = data['item_price_level_shop_rev_cnt'] / data['shop_rev_cnt']

        gender0_user_cnt = data.groupby(['gender0', 'user_id']).size().reset_index().rename(
            columns={0: 'gender0_user_cnt'})
        data = pd.merge(data, gender0_user_cnt, on=['gender0', 'user_id'], how='left')

        item_city_id_user_cnt = data.groupby(['item_city_id', 'user_id']).size().reset_index().rename(
            columns={0: 'item_city_id_user_cnt'})
        data = pd.merge(data, item_city_id_user_cnt, on=['item_city_id', 'user_id'], how='left')
        data['item_city_id_user_prob'] = data['item_city_id_user_cnt'] / data['number_user_id']

        data['price_collect'] = data['item_price_level'] + data['item_collected_level']

        item_sales_level_shop_rev_cnt = data.groupby(
            ['item_sales_level', 'shop_review_positive_rate']).size().reset_index().rename(
            columns={0: 'item_sales_level_shop_rev_cnt'})
        data = pd.merge(data, item_sales_level_shop_rev_cnt, on=['item_sales_level', 'shop_review_positive_rate'],
                        how='left')
        data['item_sales_level_shop_rev_prob'] = data['item_sales_level_shop_rev_cnt'] / data['shop_rev_cnt']

        item_sales_level_user_cnt = data.groupby(['item_sales_level', 'user_id']).size().reset_index().rename(
            columns={0: 'item_sales_level_user_cnt'})
        data = pd.merge(data, item_sales_level_user_cnt, on=['item_sales_level', 'user_id'], how='left')
        data['item_sales_level_user_prob'] = data['item_sales_level_user_cnt'] / data['number_user_id']

        data['sale_collect'] = data['item_sales_level'] + data['item_collected_level']
        item_pv_level_shop_cnt = data.groupby(['item_pv_level', 'shop_id']).size().reset_index().rename(
            columns={0: 'item_pv_level_shop_cnt'})
        data = pd.merge(data, item_pv_level_shop_cnt, on=['item_pv_level', 'shop_id'], how='left')
        data['item_pv_level_shop_prob'] = data['item_pv_level_shop_cnt'] / data['number_shop_id']

        item_city_id_shop_rev_cnt = data.groupby(
            ['item_city_id', 'shop_review_positive_rate']).size().reset_index().rename(
            columns={0: 'item_city_id_shop_rev_cnt'})
        data = pd.merge(data, item_city_id_shop_rev_cnt, on=['item_city_id', 'shop_review_positive_rate'], how='left')
        data['item_city_id_shop_rev_prob'] = data['item_city_id_shop_rev_cnt'] / data['shop_rev_cnt']

        data = data.drop(['number_item_user_id', 'item_brand_id_user_cnt', 'item_id_shop_cnt', 'shop_rev_cnt',
                          'item_price_level_shop_rev_cnt', 'item_city_id_user_cnt',
                          'item_sales_level_shop_rev_cnt', 'item_sales_level_user_cnt', 'item_pv_level_shop_cnt',
                          'item_city_id_shop_rev_cnt'], axis=1)
        return data

    def shop_fenduan(self, data):
        data['shop_score_delivery'] = data['shop_score_delivery'] * 5
        data = data[data['shop_score_delivery'] != -5]
        data['deliver_map'] = data['shop_score_delivery'].apply(self.deliver)
        data['deliver_map'] = data['deliver_map'].apply(self.deliver1)
        # del data['shop_score_delivery']
        print(data.deliver_map.value_counts())

        data['shop_score_service'] = data['shop_score_service'] * 5
        data = data[data['shop_score_service'] != -5]
        data['service_map'] = data['shop_score_service'].apply(self.service)
        data['service_map'] = data['service_map'].apply(self.service1)
        # del data['shop_score_service']
        print(data.service_map.value_counts())
        #
        data['shop_score_description'] = data['shop_score_description'] * 5
        data = data[data['shop_score_description'] != -5]
        data['de_map'] = data['shop_score_description'].apply(self.describe)
        data['de_map'] = data['de_map'].apply(self.describe1)
        # del data['shop_score_description']
        print(data.de_map.value_counts())

        data = data[data['shop_review_positive_rate'] != -1]
        data['review_map'] = data['shop_review_positive_rate'].apply(self.review)
        data['review_map'] = data['review_map'].apply(self.review1)
        print(data.review_map.value_counts())

        data['normal_shop'] = data.apply(
            lambda x: 1 if (x.deliver_map == 3) & (x.service_map == 3) & (x.de_map == 3) & (x.review_map == 3) else 0,
            axis=1)
        del data['de_map']
        del data['service_map']
        del data['deliver_map']
        del data['review_map']
        return data

    def shijian(self, data):
        data['hour_map'] = data['hour'].apply(self.map_hour)
        return data

    def timestamp_datetime(self, value):
        format = '%Y-%m-%d %H:%M:%S'
        value = time.localtime(value)
        dt = time.strftime(format, value)
        return dt

    def appendData(self, train, test):
        key = list(test)
        mergeData = pd.concat([train, test], keys=key)
        mergeData = mergeData.reset_index(drop=True)
        return mergeData
        pass
    def user_cnt_pre30min(self,data):
        def get_index(x):
            index = x[0] + np.searchsorted(temp.time[x[0]:x[1]], x[2])
            return index
        temp = data[['user_id','item_id','instance_id','time']]
        temp = temp.sort_values(['user_id', 'time'], ascending=[1, 1])
        temp['start_dates'] = temp['time'] - pd.Timedelta(minutes=30)
        left, right = np.searchsorted(temp.user_id,temp.user_id, side='left'), np.searchsorted(temp.user_id, temp.user_id, side='right')
        temp['left_index'] = left
        temp['right_index'] = right

        a=temp[['left_index','right_index','start_dates']].apply(lambda x:get_index(tuple(x)),axis=1)
        temp['start_index'] = a['left_index']
        temp['end_index'] = range(temp.shape[0])
        temp['user_pre30min_cnt'] = temp[['start_index','end_index']].apply(lambda x : x[1]-x[0],axis=1)
        temp = temp.drop(['start_dates','left_index','right_index','start_index','end_index'],axis=1)
        data = pd.merge(data,temp,'left',on=['user_id','item_id','instance_id','time'])
        # df['A'].iloc[row['start_index']:row['end_index'] + 1].sum()
        return data

    def cal_time_reduc_user_shop(self, data):
        train_origin = data
        train1 = train_origin[['context_timestamp', 'user_id', 'instance_id', 'shop_id', 'item_id']]
        train1 = train1.sort_values(['user_id', 'shop_id', 'context_timestamp'], ascending=[1, 1, 1])

        rnColumn = train1.groupby(['user_id', 'shop_id']).rank(method='min')
        train1['rnn'] = rnColumn['context_timestamp']

        # train2 = copy.deepcopy(train1)
        train1['rnn_1'] = rnColumn['context_timestamp'] - 1
        # train1.drop(['rn'],axis =1)
        # curren - last
        train2 = train1.merge(train1, how='left', left_on=['user_id', 'shop_id', 'rnn_1'],
                              right_on=['user_id', 'shop_id', 'rnn'])

        train2['time_redc_user_shop'] = train2['context_timestamp_x'] - train2['context_timestamp_y']
        train2 = train2.fillna(-1).astype('int64')
        train2 = train2.rename(columns={'instance_id_x': 'instance_id'})
        train2 = train2.rename(columns={'item_id_x': 'item_id'})
        train2 = train2.rename(columns={'shop_id_x': 'shop_id'})
        train2 = train2.rename(columns={'context_timestamp_x': 'context_timestamp'})
        train2 = train2.drop([  # 'rnn_x','rnn_y','rnn_1_x','rnn_1_y',
            'context_timestamp_y', 'instance_id_y'], axis=1)
        data = pd.merge(train_origin, train2, on=['instance_id', 'item_id', 'user_id', 'context_timestamp'], how='left')
        return data

    def cal_time_reduc_user_item(self, data):
        train_origin = data
        train1 = train_origin[['context_timestamp', 'user_id', 'instance_id', 'item_id']]
        train1 = train1.sort_values(['user_id', 'item_id', 'context_timestamp'], ascending=[1, 1, 1])

        rnColumn = train1.groupby(['user_id', 'item_id']).rank(method='min')
        train1['rnnn'] = rnColumn['context_timestamp']

        # train2 = copy.deepcopy(train1)
        train1['rnnn_1'] = rnColumn['context_timestamp'] - 1
        # train1.drop(['rn'],axis =1)
        # curren - last
        train2 = train1.merge(train1, how='left', left_on=['user_id', 'item_id', 'rnnn_1'],
                              right_on=['user_id', 'item_id', 'rnnn'])

        train2['time_redc_user_item'] = train2['context_timestamp_x'] - train2['context_timestamp_y']
        train2 = train2.fillna(-1).astype('int64')
        train2 = train2.rename(columns={'instance_id_x': 'instance_id'})
        train2 = train2.rename(columns={'item_id_x': 'item_id'})
        train2 = train2.rename(columns={'context_timestamp_x': 'context_timestamp'})
        train2 = train2.drop([  # 'rnnn_x','rn_y','rn_1_x','rn_1_y',
            'context_timestamp_y', 'instance_id_y'], axis=1)
        data = pd.merge(train_origin, train2, on=['instance_id', 'item_id', 'user_id', 'context_timestamp'], how='left')
        return data

    def cal_time_reduc(self, data):
        train_origin = data
        train1 = train_origin[['context_timestamp', 'user_id', 'instance_id', 'item_id']]
        train1 = train1.sort_values(['user_id', 'context_timestamp'], ascending=[1, 1])

        rnColumn = train1.groupby('user_id').rank(method='min')
        train1['rn'] = rnColumn['context_timestamp']

        # train2 = copy.deepcopy(train1)
        train1['rn_1'] = rnColumn['context_timestamp'] - 1
        # train1.drop(['rn'],axis =1)
        # curren - last
        train2 = train1.merge(train1, how='left', left_on=['user_id', 'rn_1'], right_on=['user_id', 'rn'])
        print(train2.shape)
        train2['time_redc'] = train2['context_timestamp_x'] - train2['context_timestamp_y']
        train2 = train2.fillna(-1).astype('int64')
        train2 = train2.rename(columns={'instance_id_x': 'instance_id'})
        train2 = train2.rename(columns={'item_id_x': 'item_id'})
        train2 = train2.rename(columns={'context_timestamp_x': 'context_timestamp'})
        user_cnt_max = train2.groupby(['user_id']).max()['rn_x'].reset_index().rename(columns={'rn_x': 'user_cnt_max'})
        train2 = pd.merge(train2,user_cnt_max,'left',on=['user_id'])
        train2['user_remain_cnt'] = train2['user_cnt_max'] - train2['rn_x']
        train2.drop(['user_cnt_max'],inplace=True,axis=1)
        data = pd.merge(train_origin, train2, on=['instance_id', 'item_id', 'user_id'], how='left')
        print(data.shape)
        return data

    def slide_cnt2(self, data):

        for d in range(18, 26):
            df1 = data[data['day'] == d]

            rnColumn_user = df1.groupby('user_id').rank(method='min')

            rnColumn_user_item = df1.groupby(['user_id','item_id']).rank(method='min')
            rnColumn_user_shop = df1.groupby(['user_id','shop_id']).rank(method='min')


            df1['user_id_order'] = rnColumn_user['context_timestamp']
            df1['user_item_id_order'] = rnColumn_user_item['context_timestamp']
            df1['user_shop_id_order'] = rnColumn_user_shop['context_timestamp']

            df2 = df1[['user_id', 'instance_id', 'item_id', 'user_id_order','user_item_id_order','user_shop_id_order']]
            if d == 18:
                Df = df2
            else:
                Df = pd.concat([Df, df2])

        data = pd.merge(data, Df, on=['user_id', 'instance_id', 'item_id'], how='left')
        data['is_trade'] = self.trainlabel
        print('list = ', list(data))
        print('initial = ', np.array(data).shape)
        # data['user_time_delta']= data[['user_id','time']].apply(user_time_delta,axis=1)

        for d in range(18, 26):  #
            df1 = data[data['day'] == d - 1]
            df2 = data[data['day'] == d]  #

            df_cvr = data[(data['day'] == d - 1) & (data['is_trade'] == 1)]

            user_item_cnt = df1.groupby(['item_id', 'user_id']).count()['instance_id'].to_dict()
            # item_trade_cnt = df1.groupby(['item_id','shop_id','is_trade']).count()['instance_id'].to_dict()

            user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
            item_cnt = df1.groupby(by='item_id').count()['instance_id'].to_dict()
            shop_cnt = df1.groupby(by='shop_id').count()['instance_id'].to_dict()

            item_cvr_cnt = df_cvr.groupby(by='item_id').count()['instance_id'].to_dict()
            user_cvr_cnt = df_cvr.groupby(by='user_id').count()['instance_id'].to_dict()

            df2['item_cvr_cnt1'] = df2['item_id'].apply(lambda x: item_cvr_cnt.get(x, 0))
            df2['user_cvr_cnt1'] = df2['user_id'].apply(lambda x: user_cvr_cnt.get(x, 0))

            df2['user_item_cnt1'] = df2[['item_id', 'user_id']].apply(lambda x: user_item_cnt.get(tuple(x), 0), axis=1)
            # df2['item_trade_cnt'] = df2[['item_id','shop_id','is_trade']].apply(lambda x: item_trade_cnt.get(tuple(x), 0),axis = 1)

            # print(df2['user_item_cnt1'].unique())
            # return
            df2['user_cnt1'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
            df2['item_cnt1'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
            df2['shop_cnt1'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))
            df2 = df2[['user_item_cnt1', 'user_cnt1', \
                       # 'item_cnt1', 'shop_cnt1',\
                       'item_cvr_cnt1', 'user_cvr_cnt1', \
                       'item_id', 'user_id', 'instance_id']]
            if d == 18:
                Df2 = df2
                print('len1 Df2 = ', np.array(Df2).shape)
            else:
                Df2 = pd.concat([df2, Df2])
                print('len2 Df2 = ', np.array(Df2).shape)

        # data = Df2
        print('len3 data = ', np.array(data).shape)
        print('first list data = ', list(data))

        data = pd.merge(data, Df2, on=['instance_id', 'item_id', 'user_id'], how='left')
        print('after1 = ', np.array(data).shape)
        print('cnt')
        for d in range(18, 26):
            #
            df1 = data[data['day'] < d]
            df2 = data[data['day'] == d]

            df_cvr = data[(data['day'] < d) & (data['is_trade'] == 1)]

            # item_trade_cnt = df1.groupby(['item_id','shop_id','is_trade']).count()['instance_id'].to_dict()
            user_item_cnt = df1.groupby(['item_id', 'user_id']).count()['instance_id'].to_dict()

            user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
            item_cnt = df1.groupby(by='item_id').count()['instance_id'].to_dict()
            shop_cnt = df1.groupby(by='shop_id').count()['instance_id'].to_dict()

            item_cvr_cnt = df_cvr.groupby(by='item_id').count()['instance_id'].to_dict()
            user_cvr_cnt = df_cvr.groupby(by='user_id').count()['instance_id'].to_dict()

            df2['item_cvr_cntx'] = df2['item_id'].apply(lambda x: item_cvr_cnt.get(x, 0))
            df2['user_cvr_cntx'] = df2['user_id'].apply(lambda x: user_cvr_cnt.get(x, 0))
            df2['user_item_cntx'] = df2[['item_id', 'user_id']].apply(lambda x: user_item_cnt.get(tuple(x), 0), axis=1)

            # return
            df2['user_cntx'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))

            df2 = df2[['user_item_cntx', 'user_cntx',
                       # 'user_shop_cntx','user_shop_cvr_cntx',
                       # 'user_category_cntx', 'user_item_cvr_cntx','user_category_cvr_cntx',
                       'item_cvr_cntx', 'user_cvr_cntx', \
                       'item_id', 'user_id', 'instance_id']]

            if d == 18:
                Df2 = df2
                print('len1-1 Df2 = ', np.array(Df2).shape)
            else:
                Df2 = pd.concat([df2, Df2])
                print('len2-1 Df2 = ', np.array(Df2).shape)

        data = pd.merge(data, Df2, on=['instance_id', 'item_id', 'user_id'], how='left')

        print('len3-1 Df2 = ', np.array(data).shape)
        print('second list data = ', list(data))

        # print("pre hour")
        print('after2 = ', np.array(data).shape)

        print('two day cnt')
        for d in range(18, 26):
            df1 = data[(data['day'] >= d - 2) & (data['day'] < d)]
            df2 = data[data['day'] == d]
            # item_trade_cnt = df1.groupby(['item_id','shop_id','is_trade']).count()['instance_id'].to_dict()
            user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
            item_cnt = df1.groupby(by='item_id').count()['instance_id'].to_dict()
            shop_cnt = df1.groupby(by='shop_id').count()['instance_id'].to_dict()
            user_item_cnt = df1.groupby(['item_id', 'user_id']).count()['instance_id'].to_dict()

            df2['user_item_cnt2'] = df2[['item_id', 'user_id']].apply(lambda x: user_item_cnt.get(tuple(x), 0), axis=1)
            df2['user_cnt2'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
            df2['item_cnt2'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
            df2['shop_cnt2'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))
            df2 = df2[['user_item_cnt2', 'user_cnt2', 'item_cnt2', 'shop_cnt2', \
                       'item_id', 'user_id', 'instance_id']]

            if d == 18:
                Df2 = df2
                print('len1 Df2 = ', np.array(Df2).shape)
            else:
                Df2 = pd.concat([df2, Df2])
                print('len2 Df2 = ', np.array(Df2).shape)
        print('len3 Df2 = ', np.array(Df2).shape)
        print('list Df2', list(Df2))

        data = pd.merge(data, Df2, on=['instance_id', 'item_id', 'user_id'], how='left')

        print('data shape', np.array(data).shape)
        return data

    def convert_data(self, data):
        data['time'] = pd.to_datetime(data.context_timestamp.apply(self.timestamp_datetime))
        data['day'] = data.time.dt.day
        data['hour'] = data.time.dt.hour
        data['minute'] = data.time.dt.minute
        item_category_list_2 = pd.DataFrame([int(i.split(';')[1]) for i in data.item_category_list])
        data['item_category_list_2'] = item_category_list_2

        lbl = preprocessing.LabelEncoder()
        for col in ['item_id', 'item_brand_id', 'item_city_id', 'shop_id', 'user_id']:
            data[col] = lbl.fit_transform(data[col])
        ### external time data
        user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
        data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
        user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
            columns={0: 'user_query_day_hour'})
        data = pd.merge(data, user_query_day_hour, 'left', on=['user_id', 'day', 'hour'])

        ###
        '''
        item_id_frequence = data.groupby([ 'item_id']).size().reset_index().rename(columns={0: 'item_id_frequence'})
        item_id_frequence=item_id_frequence/(data.shape[0])
        data = pd.merge(data, item_id_frequence, 'left', on=['item_id'])
        '''
        ###
        # num_user_minute = data.groupby(['user_id','day','minute']).size().reset_index().rename(columns = {0:'num_user_day_minute'})
        # data = pd.merge(data, num_user_minute,'left',on = ['user_id','day','minute'])

        day_user_item_id = data.groupby(['day', 'user_id', 'item_id']).size().reset_index().rename(
            columns={0: 'day_user_item_id'})
        data = pd.merge(data, day_user_item_id, 'left', on=['day', 'user_id', 'item_id'])

        day_hour_minute_user_item_id = data.groupby(
            ['day', 'hour', 'minute', 'user_id', 'item_id']).size().reset_index().rename(
            columns={0: 'day_hour_minute_user_item_id'})
        data = pd.merge(data, day_hour_minute_user_item_id, 'left', on=['day', 'hour', 'minute', 'user_id', 'item_id'])

        number_day_hour_item_id = data.groupby(['day', 'hour', 'item_id']).size().reset_index().rename(
            columns={0: 'number_day_hour_item_id'})
        data = pd.merge(data, number_day_hour_item_id, 'left', on=['day', 'hour', 'item_id'])

        item_user_id = data.groupby(['item_id', 'user_id']).size().reset_index().rename(columns={0: 'item_user_id'})
        data = pd.merge(data, item_user_id, 'left', on=['item_id', 'user_id'])
        # new feature

        item_category_city_id = data.groupby(['item_category_list', 'item_city_id']).size().reset_index().rename(
            columns={0: 'item_category_city_id'})
        data = pd.merge(data, item_category_city_id, 'left', on=['item_category_list', 'item_city_id'])

        item_category_sales_level = data.groupby(
            ['item_category_list', 'item_sales_level']).size().reset_index().rename(
            columns={0: 'item_category_sales_level'})
        data = pd.merge(data, item_category_sales_level, 'left', on=['item_category_list', 'item_sales_level'])

        item_category_price_level = data.groupby(
            ['item_category_list', 'item_price_level']).size().reset_index().rename(
            columns={0: 'item_category_price_level'})
        data = pd.merge(data, item_category_price_level, 'left', on=['item_category_list', 'item_price_level'])

        item_ID_sales_level = data.groupby(['item_id', 'item_sales_level']).size().reset_index().rename(
            columns={0: 'item_ID_sales_level'})
        data = pd.merge(data, item_ID_sales_level, 'left', on=['item_id', 'item_sales_level'])

        item_ID_collected_level = data.groupby(['item_id', 'item_collected_level']).size().reset_index().rename(
            columns={0: 'item_ID_collected_level'})
        data = pd.merge(data, item_ID_collected_level, 'left', on=['item_id', 'item_collected_level'])

        ### dangerous feature
        number_user_id = data.groupby(['user_id']).size().reset_index().rename(columns={0: 'number_user_id'})
        data = pd.merge(data, number_user_id, 'left', on=['user_id'])

        number_shop_id = data.groupby(['shop_id']).size().reset_index().rename(columns={0: 'number_shop_id'})
        data = pd.merge(data, number_shop_id, 'left', on=['shop_id'])

        ### label encoder

        for i in range(5):
            data['predict_category_property' + str(i)] = lbl.fit_transform(data['predict_category_property'].map(
                lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))

        for i in range(1, 3):
            data['item_category_list' + str(i)] = lbl.fit_transform(data['item_category_list'].map(
                lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))  # item_category

        for i in range(10):
            data['item_property_list' + str(i)] = lbl.fit_transform(data['item_property_list'].map(
                lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))

            # data['context_page0'] = data['context_page_id'].apply(
            #       lambda x: 1 if x == 4001 | x == 4002 | x == 4003 | x == 4004 | x == 4007  else 2)

        data['gender0'] = data['user_gender_id'].apply(lambda x: x + 1 if x == -1 else x)
        data['age0'] = data['user_age_level'].apply(lambda x: 1003 if x == -1  else x)
        data['occupation0'] = data['user_occupation_id'].apply(lambda x: 2005 if x == -1  else x)
        data['star0'] = data['user_star_level'].apply(lambda x: 3006 if x == -1 else x)
        # new
        # number_item_user_id = data.groupby(['item_id','user_id']).size().reset_index().rename(columns={0: 'number_item_user_id'})
        # data = pd.merge(data, number_item_user_id, 'left',on=['item_id','user_id'])

        number_item_brand_positive_rate = data.groupby(
            ['item_brand_id', 'shop_review_positive_rate']).size().reset_index().rename(
            columns={0: 'number_item_brand_positive_rate'})
        data = pd.merge(data, number_item_brand_positive_rate, 'left',
                        on=['item_brand_id', 'shop_review_positive_rate'])

        number_item_brand_shop_star = data.groupby(['item_brand_id', 'shop_star_level']).size().reset_index().rename(
            columns={0: 'number_item_brand_shop_star'})
        data = pd.merge(data, number_item_brand_shop_star, 'left', on=['item_brand_id', 'shop_star_level'])

        number_item_city_pv_level = data.groupby(['item_city_id', 'item_pv_level']).size().reset_index().rename(
            columns={0: 'number_item_city_pv_level'})
        data = pd.merge(data, number_item_city_pv_level, 'left', on=['item_city_id', 'item_pv_level'])

        number_item_city_user_id = data.groupby(['item_city_id', 'user_id']).size().reset_index().rename(
            columns={0: 'number_item_city_user_id'})
        data = pd.merge(data, number_item_city_user_id, 'left', on=['item_city_id', 'user_id'])

        number_item_price_sales_level = data.groupby(
            ['item_price_level', 'item_sales_level']).size().reset_index().rename(
            columns={0: 'number_item_price_sales_level'})
        data = pd.merge(data, number_item_price_sales_level, 'left', on=['item_price_level', 'item_sales_level'])

        number_predict_category_sales_level = data.groupby(
            ['predict_category_property', 'item_sales_level']).size().reset_index().rename(
            columns={0: 'number_predict_category_sales_level'})
        data = pd.merge(data, number_predict_category_sales_level, 'left',
                        on=['predict_category_property', 'item_sales_level'])

        number_collected_shop_id = data.groupby(['item_collected_level', 'shop_id']).size().reset_index().rename(
            columns={0: 'number_collected_shop_id'})
        data = pd.merge(data, number_collected_shop_id, 'left', on=['item_collected_level', 'shop_id'])

        print('item_category_list_ing')
        for i in range(3):
            data['category_%d' % (i)] = data['item_category_list'].apply(
                lambda x: x.split(";")[i] if len(x.split(";")) > i else " ")

        ###
        print('item_property_list_ing')
        for i in range(3):
            data['property_%d' % (i)] = data['item_property_list'].apply(
                lambda x: x.split(";")[i] if len(x.split(";")) > i else " ")
        ###
        print('predict_category_property_ing_0')
        for i in range(3):
            data['predict_category_%d' % (i)] = data['predict_category_property'].apply(
                lambda x: str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else " ")
        for i in ['item_id','shop_id','day','context_page_id']:
           temp=data.groupby('user_id').nunique()[i].reset_index().rename(columns={i:'number_'+i+'_query_user'})
           data=pd.merge(data,temp,'left',on='user_id')
        return data

    def datafilter(self, ad_data):
        basic_data = ad_data[['instance_id']]
        ad_information = ad_data[
            ['item_id', 'item_category_list', 'item_brand_id', 'item_city_id', 'item_price_level','item_property_list',
             'item_sales_level', 'item_collected_level', 'item_pv_level']]
        user_information = ad_data[
            ['user_id', 'user_age_level', 'user_star_level', 'user_occupation_id']]  # ,'user_gender_id',
        text_information = ad_data[['context_id', 'context_timestamp', 'context_page_id', 'predict_category_property']]
        shop_information = ad_data[
            ['shop_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service',
             'shop_score_delivery', 'shop_score_description']]
        external_information = ad_data[
            ['time', 'day', 'hour', 'minute', 'user_query_day', 'user_query_day_hour', 'day_user_item_id', \
             'day_hour_minute_user_item_id',
             'number_day_hour_item_id', 'number_user_id', 'number_shop_id', \
             'item_category_list_2', 'item_user_id', 'item_category_city_id', 'item_category_sales_level', \
             'item_ID_sales_level', 'item_ID_collected_level', 'item_category_price_level', \
             'predict_category_property0', 'predict_category_property1', 'predict_category_property2', \
             'predict_category_property3', 'predict_category_property4', 'item_category_list1', \
             'item_category_list2', 'item_property_list0', 'item_property_list1', 'item_property_list2', \
             'item_property_list3', 'item_property_list4', 'item_property_list5', 'item_property_list6', \
             'item_property_list7', 'item_property_list8', 'item_property_list9', 'gender0', 'age0', \
             'occupation0', 'star0', 'number_item_brand_positive_rate', 'number_item_brand_shop_star', \
             'number_item_city_pv_level', 'number_item_city_user_id', 'number_item_price_sales_level', \
             'number_predict_category_sales_level', 'number_collected_shop_id'#,'shop_score_delivery_round',
             # 'number_item_id_query_user' #,'number_shop_id_query_user','number_day_query_user' #,'number_context_page_id_query_user'
             ]]

        result = pd.concat(
            [basic_data, ad_information, user_information, text_information, shop_information, external_information],
            axis=1)
        '''
        important feature next line
        '''

        print('---------------------------')
        # ItemCVR = concatDayCVR()
        # result = pd.merge(result, ItemCVR, on=['instance_id','item_id','user_id'], how = 'left')

        '''
        I saved Bayes_smooth result for save time, you can run itemIDBayesSmooth.py, userIDBayesSmooth.py etc to get the result. 
        '''
        # Item_Bayes = np.load('./datasets/Item_Bayes.npy')
        # # Brand_Bayes=np.load('./datasets/Brand_id_Bayes.npy')
        # Shop_Bayes = np.load('./datasets/Shop_Bayes.npy')
        # # Hour_Bayes = np.load('./datasets/Item_pv_levelBayesPH.npy')
        # User_Bayes = np.load('./datasets/UserBayesPH.npy')
        # result['Item_Bayes'] = Item_Bayes
        # result['Brand_Bayes'] = Brand_Bayes
        # result['Shop_Bayes'] = Shop_Bayes
        # result['User_Bayes'] = concatUserDayCVR()
        # result['Hour_Bayes'] = Hour_Bayes
        # result['itemCVR'] = concatItemDayCVR()

        # result = zuhe(result)
        # result = item(result)
        # result = user(result)
        # result = user_item(result)
        # result = user_shop(result)
        # result = shop_item(result)
        # result = self.zuhe_feature(result)

        result = self.slide_cnt2(result)
        result = self.cal_time_reduc_user_item(result)
        result = self.cal_time_reduc_user_shop(result)

        result = self.cal_time_reduc(result)
        result = self.extra_feature(result)
        # '''
        # result = zuhe(result)
        # result = item(result)
        # result = user(result)

        # result = user_shop(result)
        # result = user_item(result)
        # result = shop_item(result)
        # result = self.feature_filter(result)
        # '''
        result = result.drop(
            ['item_category_list', 'item_property_list', 'predict_category_property', 'time',
             'instance_id'], axis=1)

        if self.evalOrTest == 'eval':
            result = result.iloc[0:self.trainLen, :]
            target = self.trainlabel
            return result, target

        if self.evalOrTest == 'test':
            result = result.iloc[self.trainLen:, :]
            return result

    '''
    new preprocess function will be added to datafilter function
    '''

    # this is log likelihood loss
    def logregobj(self, y_hat, dtrain):
        y = dtrain.get_label()
        p = y_hat
        beta = 0.1
        grad = p * (beta + y - beta * y) - y
        hess = p * (1 - p) * (beta + y - beta * y)
        return grad, hess

    def logAverage(self, loglist):
        num = len(loglist)
        val = len(loglist[0])
        result = []

        for j in range(val):
            cal = 0
            for i in range(num):
                tmplist = loglist[i]
                cal = cal + tmplist[j]
                cal = cal / num;
                result.append(cal)
        return result

    def gbdtparamSetting(self):
        param = {
            'n_estimators': self.treeNum,
            'learning_rate': 0.05,
            'max_depth': 3,
            'random_state': 0,
            'loss': 'deviance'
        }
        return param

    def xgbparamSetting(self):
        param = {
            'learning_rate': 0.05,
            'eta': 0.4,
            'max_depth': 3,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'alpha': 1,
            # 'lambda' : 0.1,
            'nthread': 4,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        return param

    def LGBMparamSetting(self):
        param = {
            'learning_rate': 0.01,
            'num_leaves': 32,
            # 'eta' : 0.4,
            'subsample': 0.35,
            'colsample_bytree': 0.3,
            'nthread': 4,
            # 'lambda_l1' : 0.1,
            'objective': 'binary',
            'metric': 'binary_logloss'
        }
        return param

    def gbdtEval(self, dataTrain, labelTrain, dataTest, labelTest, foldNum):
        modelProcess = 'gbdtProcess'
        self.writelog0(modelProcess)
        param = self.gbdtparamSetting()
        self.writelog1(param)

        # num_boost_round = self.treeNum
        clf = GradientBoostingClassifier(n_estimators=param['n_estimators'], \
                                         learning_rate=param['learning_rate'], \
                                         max_depth=param['max_depth'], \
                                         random_state=0)
        gbdt_model = clf.fit(dataTrain, labelTrain)
        result = gbdt_model.predict_proba(dataTest)[:, 1]
        loglossList = log_loss(labelTest, result)
        joblib.dump(clf, 'gbdtModel' + str(foldNum))
        return loglossList

    def lgbEval(self, dataTrain, labelTrain, dataTest, labelTest, foldNum):
        modelProcess = 'lgbProcess'
        dtrain = lgb.Dataset(dataTrain, label=labelTrain)
        dtest = lgb.Dataset(dataTest, label=labelTest)
        process = dict()
        param = self.LGBMparamSetting()

        num_boost_round = self.treeNum
        bst = lgb.train(param, dtrain, num_boost_round, valid_sets=dtest, early_stopping_rounds=500, \
                        evals_result=process)
        bst.save_model('lgbModel' + str(foldNum))
        # print(process)
        loglossList = process['valid_0']['binary_logloss']
        feature_score = self.feature_sort(bst)
        with open("feature_score.txt", "a") as f:
            f.write(str(param) + '\n')
            f.write(str(bst.best_score) + str(bst.best_iteration) + '\n')
            f.write(str(feature_score) + '\n')
        return loglossList

    def feature_sort(self, bst):
        score = bst.feature_importance()
        name = bst.feature_name()
        return sorted(zip(name, score), key=lambda a: a[1], reverse=True)

    def xgbEval(self, dataTrain, labelTrain, dataTest, labelTest, foldNum):
        # print('finalLabel', set(list(np.array(labelTrain))) )
        modelProcess = 'xgbProcess'
        # self.writelog0(modelProcess)
        dtrain = xgb.DMatrix(data=dataTrain.values, label=labelTrain.values)
        dtest = xgb.DMatrix(data=dataTest.values, label=labelTest.values)

        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        progress = dict()
        param = self.xgbparamSetting()
        # self.writelog1(param)

        num_boost_round = self.treeNum
        bst = xgb.train(param, dtrain, num_boost_round, watchlist, early_stopping_rounds=150, \
                        evals_result=progress)
        # self.treeNum = num_boost_round
        bst.save_model('xgbModel' + str(foldNum))
        xgb.plot_importance(bst)
        loglossList = progress['test']['logloss']
        # self.writelog2(loglossList)
        return loglossList

    def modelBoostEval(self, data, target):
        kf = KFold(len(target), n_folds=self.n_fold, random_state=self.random_seed)
        foldNum = 1
        for trainIndex, testIndex in kf:
            print('K-fold ', foldNum)
            dataTrain, dataTest, labelTrain, labelTest = data.iloc[trainIndex], data.iloc[testIndex], target.iloc[
                trainIndex], target.iloc[testIndex]
            self.catBoostEval(dataTrain, labelTrain, dataTest, labelTest, foldNum)
            foldNum = foldNum + 1

    '''
    plan to add evalmodelchice
    '''

    def modelFiveFoldEval(self, data, target):

        kf = KFold(len(target), n_folds=self.n_fold, random_state=self.random_seed)
        foldNum = 1
        loglossList = []
        avglogloss = 0
        for trainIndex, testIndex in kf:
            print('K-fold ', foldNum)
            dataTrain, dataTest, labelTrain, labelTest = data.iloc[trainIndex], data.iloc[testIndex], target.iloc[
                trainIndex], target.iloc[testIndex]

            # logList = self.xgbEval(dataTrain, labelTrain, dataTest, labelTest, foldNum)
            logList = self.gbdtEval(dataTrain, labelTrain, dataTest, labelTest, foldNum)
            # logList = self.lgbEval(dataTrain, labelTrain, dataTest, labelTest, foldNum)
            tmplogloss = np.min(np.array(logList))
            avglogloss = avglogloss + tmplogloss
            loglossList.append(logList)
            foldNum = foldNum + 1
        avglogloss = avglogloss / self.n_fold
        self.writelog3(avglogloss)
        return self.logAverage(loglossList)

    def modelDayEval(self, data, target):

        data['target'] = np.array(target)

        loglossList = []
        avglogloss = 0
        foldNum = 1

        day24 = data[data['day'] == 24]

        day18_23 = data[data['day'] < 24]
        day19_23 = day18_23[data['day'] >= 18]
        # jius1print('18~23',list(day18_23))
        dataTrain = day19_23.drop(['target'], axis=1)
        labelTrain = day19_23['target']

        dataTest = day24.drop(['target'], axis=1)
        labelTest = day24['target']

        # logList = self.xgbEval(dataTrain, labelTrain, dataTest, labelTest, foldNum)
        logList = self.lgbEval(dataTrain, labelTrain, dataTest, labelTest, foldNum)
        tmplogloss = np.min(np.array(logList))
        avglogloss = avglogloss + tmplogloss
        loglossList.append(logList)
        # foldNum = foldNum + 1
        avglogloss = avglogloss
        return self.logAverage(loglossList)
        pass

    def modelTrain(self, data, target, modelChoice):
        if modelChoice == 'XGBModel' or modelChoice == 'all':
            dtrain = xgb.DMatrix(data=data.values, label=target.values)
            progress = dict()
            param = self.xgbparamSetting()
            num_round = self.XGBbestNumRounds
            bst = xgb.train(param, dtrain, num_round, evals_result=progress)
            bst.save_model('xgbModelFinal')

        if modelChoice == 'LGBMmodel' or modelChoice == 'all':
            dtrain = lgb.Dataset(data=data.values, label=target.values)
            progress = dict()
            param = self.LGBMparamSetting()
            num_round = self.LGBMbestNumRounds
            bst = lgb.train(param, dtrain, num_round, evals_result=progress)
            bst.save_model('lgbModelFinal')

    def modelTest(self, data, modelChoice):
        writefileName = 'result/20180422.csv'
        if modelChoice == 'XGBModel':
            # print('11')
            model = xgb.Booster(model_file='xgbModelFinal')
            preds = model.predict(xgb.DMatrix(data.values))
            sub = pd.DataFrame()
            sub['instance_id'] = self.testInstanceID
            sub['predicted_score'] = preds
            sub.to_csv(writefileName, sep=" ", index=False, line_terminator='\r')
            return preds

        if modelChoice == 'LGBMmodel':
            model = lgb.Booster(model_file='lgbModelFinal')
            preds = model.predict(data.values)
            sub = pd.DataFrame()
            sub['instance_id'] = self.testInstanceID
            sub['predicted_score'] = preds
            sub.to_csv(writefileName, sep=" ", index=False, line_terminator='\r')
            return preds

        if modelChoice == 'all':
            # print('11')
            XGBmodel = xgb.Booster(model_file='xgbModelFinal')
            XGBpreds = XGBmodel.predict(xgb.DMatrix(data.values))
            LGBMmodel = lgb.Booster(model_file='lgbModelFinal')
            LGBMpreds = LGBMmodel.predict(data.values)
            preds = 0.5 * XGBpreds + 0.5 * LGBMpreds
            sub = pd.DataFrame()
            sub['instance_id'] = self.testInstanceID
            sub['predicted_score'] = preds
            sub.to_csv(writefileName, sep=" ", index=False, line_terminator='\r')
            return preds

    # extract onehot feature

    def getGBDTFeatureTest(self, data):
        data = np.array(data)
        featureList = np.zeros((len(data), self.treeNum * pow(2, (self.xgbparamSetting()['max_depth'] + 1))))
        grd_enc = OneHotEncoder(pow(2, self.xgbparamSetting()['max_depth'] + 1))
        model = xgb.Booster(model_file='xgbModelFinal')
        GBDTfeature = model.predict(xgb.DMatrix(data), ntree_limit=self.nTreeLimit, pred_leaf=True)
        oneHotFeature = grd_enc.fit_transform(GBDTfeature.reshape(-1, 1))
        oneHotFeature = oneHotFeature.toarray()
        featureList = oneHotFeature.reshape(len(data), -1)
        return featureList

    def convert2onehot(self, data):
        enc = preprocessing.OneHotEncoder()
        print(data[0])
        data = enc.fit_transform(data).toarray()

        # data1 = enc.transform(data[0].reshape(1,-1)).toarray()
        print(len(data[0]))
        return data
        pass

    def featureAppend(self, data, featureList):
        # data = np.array(data)
        # appendData = np.hstack((data, featureList))
        return featureList

def evalModel():
    model = solution('eval')
    ad_data = model.readData()
    ad_data = model.convert_data(ad_data)
    data, target = model.datafilter(ad_data)
    # model.writeFeature(data)
    # model.modelFiveFoldEval(data, target)
    model.modelDayEval(data, target)
    return data


def train(modelChoice):
    # train process
    model = solution('eval')
    ad_data = model.readData()
    ad_data = model.convert_data(ad_data)
    data, target = model.datafilter(ad_data)
    model.modelTrain(data, target, modelChoice)

def test(modelChoice):
    # test process
    model = solution('test')
    ad_data = model.readData()
    ad_data = model.convert_data(ad_data)
    data = model.datafilter(ad_data)
    preds = model.modelTest(data, modelChoice)
    return preds


def getTrainFeature():
    model = solution('eval')
    ad_data = model.readData()
    ad_data = model.convert_data(ad_data)
    data, target = model.datafilter(ad_data)
    featurelist = model.getGBDTFeatureTrainOnehot(data)

    appendData = model.featureAppend(data, featurelist)
    # np.savetxt("feature_onehot_only_20180321.txt", featurelist);
    np.savetxt('GBDTTrainFeature.txt', appendData)
    pass

# Run train before getTestFeature
# This program is depended on xgbModelFinal
# with trainModel to produce GBDT feature for testing

def getTestFeature():
    model = solution('test')
    ad_data = model.readData()
    ad_data = model.convert_data(ad_data)
    data, target = model.datafilter(ad_data)
    featurelist = model.getGBDTFeatureTest(data)

    np.savetxt('GBDTTestFeature.txt', featurelist)
    pass


if __name__ == '__main__':
    '''
    train select model from 'LGBMmodel' 'XGBModel' 'all'
    '''
    # train('all')
    '''
    test select model from 'LGBMmodel' 'XGBModel' 'all'
    '''
    # preds = test('all')
    '''
    get GBDTTrain feature for stage2 logistics regression eval or Train
    '''
    # getTrainFeature()
    '''
     get GBDTTest feature for stage2 logistics regression test submit result
    '''
    # getTestFeature()
    '''
    eval
    '''
    ad_data = evalModel()






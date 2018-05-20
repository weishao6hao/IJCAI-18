# -*- coding: utf-8 -*-

#挖掘转化率平滑的代码

import scipy.special as special
from collections import Counter
import pandas as pd 
import time
import numpy as np

class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            print(new_alpha,new_beta,i)
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0
        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i]+alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i]-clks[i]+beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(numerator_alpha/denominator), beta*(numerator_beta/denominator)

def readData():
    test1 = pd.read_csv('datasets/round1_ijcai_18_test_a_20180301.txt', sep = ' ')
    test2 = pd.read_table('datasets/round1_ijcai_18_test_b_20180418.txt',sep=' ')
    test=test1.append(test2)
    train = pd.read_csv('datasets/round1_ijcai_18_train_20180301.txt', sep = ' ')
    train.drop_duplicates(inplace = True)
    key = list( test )
    print(key)
    mergeData = pd.concat([train, test], keys = key)
    mergeData = mergeData.reset_index( drop = True )
    mergeData['time'] = mergeData.context_timestamp.apply(timestamp_datetime)
    mergeData['day'] = mergeData.time.apply(lambda x: int(x[8:10]))
    mergeData['hour'] = mergeData.time.apply(lambda x: int(x[11:13]))
    mergeData['minute'] = mergeData.time.apply(lambda x: int(x[14:16]))
    return mergeData

def timestamp_datetime(value):
        format = '%Y-%m-%d %H:%M:%S'
        value = time.localtime(value)
        dt = time.strftime(format, value)
        return dt
    


def Day19smooth_UserID():
        df_train = readData()

        df_train_day_pre = df_train[df_train['day'] == 18]
        df_train_day_current = df_train[df_train['day'] == 19]
        pos_all_list=list(set(df_train_day_current.user_id.values)) 
        
        # -------------- day 19 -----------day 18 for alpha beta------------------------------
        print('开始统计pos平滑')
        #### calculate the alpha and beta with day18 data for day19
        bs = BayesianSmoothing(0.836310944829, 40.4025976056)    
        dic_i = dict(Counter(df_train_day_pre.user_id.values))
        dic_cov = dict(Counter(df_train_day_pre[df_train_day_pre['is_trade'] == 1].user_id.values))  
        l = list(set(df_train_day_pre.user_id.values))     
        I = []
        C = []
        for posID in l:
            I.append(dic_i[posID])
        for posID in l:
            if posID not in dic_cov:
                C.append(0)
            else:
                C.append(dic_cov[posID])        
        print('开始平滑操作')           
        bs.update(I, C, 100000, 0.0001)
        print(bs.alpha, bs.beta)  
        print('构建平滑转化率')
        #dic_i_current = dict(Counter(df_train_day_current.item_id.values))
        #dic_cov_current = dict(Counter(df_train_day_current[df_train_day_current['is_trade'] == 1].item_id.values)) 
        dic_PH={}
        for pos in pos_all_list:
            if pos not in dic_i:
                dic_PH[pos] = (bs.alpha)/(bs.alpha + bs.beta)
            elif pos not in dic_cov:
                dic_PH[pos] = (bs.alpha)/(dic_i[pos] + bs.alpha + bs.beta)
            else:
                dic_PH[pos] = (dic_cov[pos] + bs.alpha)/(dic_i[pos] + bs.alpha+bs.beta)   
        #df_out=pd.DataFrame({'item_id':list(dic_PH.keys()),
         #                'PH_pos':list(dic_PH.values())})
    
        #df_out.to_csv('data/feature/PL_pos.csv',index=False)
        print('开始复制保存')
        ### !!! DAY19_data is data to merge
        DAY19_data = df_train_day_current[['instance_id','user_id','item_id']]
        user_ID_all = df_train_day_current['user_id']
        PH_user_id = []

        for i in user_ID_all:
             PH_user_id.append(dic_PH[i])
             
        DAY19_data['SmoothUserCVR'] = PH_user_id
    
        return DAY19_data

def Day20smooth_UserID():   
        df_train = readData()
        #pos_all_list=list(set(df_train.item_id.values))   
        df_train_day_pre = df_train[(df_train['day'] >= 18) & (df_train['day'] <= 19)]
        df_train_day_current = df_train[df_train['day'] == 20]
        pos_all_list=list(set(df_train_day_current.user_id.values)) 
        # -------------- day 20 -----------day 18 19 for alpha beta------------------------------
        #-------------------------------------
        print('开始统计pos平滑')
        #### calculate the alpha and beta with day18 data for day19
        bs = BayesianSmoothing(0.853476687842, 41.455709224)    
        dic_i = dict(Counter(df_train_day_pre.user_id.values))
        dic_cov = dict(Counter(df_train_day_pre[df_train_day_pre['is_trade'] == 1].user_id.values))  
        l = list(set(df_train_day_pre.user_id.values))     
        I = []
        C = []
        for posID in l:
            I.append(dic_i[posID])
        for posID in l:
            if posID not in dic_cov:
                C.append(0)
            else:
                C.append(dic_cov[posID])        
        print('开始平滑操作')           
        bs.update(I, C, 100000, 0.0001)
        print(bs.alpha, bs.beta) 
        
        
        print('构建平滑转化率')
        #dic_i_current = dict(Counter(df_train_day_current.item_id.values))
        #dic_cov_current = dict(Counter(df_train_day_current[df_train_day_current['is_trade'] == 1].item_id.values)) 
        dic_PH={}
        for pos in pos_all_list:
            if pos not in dic_i:
                dic_PH[pos] = (bs.alpha)/(bs.alpha + bs.beta)
            elif pos not in dic_cov:
                dic_PH[pos] = (bs.alpha)/(dic_i[pos] + bs.alpha + bs.beta)
            else:
                dic_PH[pos] = (dic_cov[pos] + bs.alpha)/(dic_i[pos] + bs.alpha+bs.beta)  
        #df_out=pd.DataFrame({'item_id':list(dic_PH.keys()),
         #                'PH_pos':list(dic_PH.values())})
    
        #df_out.to_csv('data/feature/PL_pos.csv',index=False)
        print('开始复制保存')
        ### !!! DAY19_data is data to merge
        DAY20_data = df_train_day_current[['instance_id','user_id','item_id']]
        user_ID_all = df_train_day_current['user_id']
        
        PH_user_ID = []
  
        for i in user_ID_all:
             PH_user_ID.append(dic_PH[i])
             
        DAY20_data['SmoothUserCVR'] = PH_user_ID
        
        return DAY20_data
    
def Day21smooth_UserID():   
        df_train = readData()
        #pos_all_list=list(set(df_train.item_id.values))   
        df_train_day_pre = df_train[(df_train['day'] >= 18) & (df_train['day'] <= 20)]
        df_train_day_current = df_train[df_train['day'] == 21]
        pos_all_list=list(set(df_train_day_current.user_id.values)) 
        # -------------- day 20 -----------day 18 19 for alpha beta------------------------------
        #-------------------------------------
        print('开始统计pos平滑')
        #### calculate the alpha and beta with day18 data for day19
        bs = BayesianSmoothing(1.12520231044, 55.2939856049)       
        dic_i = dict(Counter(df_train_day_pre.user_id.values))
        dic_cov = dict(Counter(df_train_day_pre[df_train_day_pre['is_trade'] == 1].user_id.values))  
        l = list(set(df_train_day_pre.user_id.values))     
        I = []
        C = []
        for posID in l:
            I.append(dic_i[posID])
        for posID in l:
            if posID not in dic_cov:
                C.append(0)
            else:
                C.append(dic_cov[posID])        
        print('开始平滑操作')           
        bs.update(I, C, 10000, 0.0001)
        print(bs.alpha, bs.beta) 
        
        
        print('构建平滑转化率')
        #dic_i_current = dict(Counter(df0878226458_train_day_current.item_id.values))
        #dic_cov_current = dict(Counter(df_train_day_current[df_train_day_current['is_trade'] == 1].item_id.values)) 
        dic_PH={}
        for pos in pos_all_list:
            if pos not in dic_i:
                dic_PH[pos] = (bs.alpha)/(bs.alpha + bs.beta)
            elif pos not in dic_cov:
                dic_PH[pos] = (bs.alpha)/(dic_i[pos] + bs.alpha + bs.beta)
            else:
                dic_PH[pos] = (dic_cov[pos] + bs.alpha)/(dic_i[pos] + bs.alpha+bs.beta) 
        #df_out=pd.DataFrame({'item_id':list(dic_PH.keys()),
         #                'PH_pos':list(dic_PH.values())})
    
        #df_out.to_csv('data/feature/PL_pos.csv',index=False)
        print('开始复制保存')
        ### !!! DAY19_data is data to merge
        DAY21_data = df_train_day_current[['instance_id','user_id','item_id']]
        user_ID_all = df_train_day_current['user_id']
        
        PH_user_id = []
  
        for i in user_ID_all:
             PH_user_id.append(dic_PH[i])
             
        DAY21_data['SmoothUserCVR'] = PH_user_id
        
        return DAY21_data
    
def Day22smooth_UserID():   
        df_train = readData()
        #pos_all_list=list(set(df_train.item_id.values))   
        df_train_day_pre = df_train[(df_train['day'] >= 18) & (df_train['day'] <= 21)]
        df_train_day_current = df_train[df_train['day'] == 22]
        pos_all_list=list(set(df_train_day_current.user_id.values)) 
        # -------------- day 20 -----------day 18 19 for alpha beta------------------------------
        #-------------------------------------
        print('开始统计pos平滑')
        #### calculate the alpha and beta with day18 data for day19
        bs = BayesianSmoothing(1.01518229823, 49.9983206048)
        dic_i = dict(Counter(df_train_day_pre.user_id.values))
        dic_cov = dict(Counter(df_train_day_pre[df_train_day_pre['is_trade'] == 1].user_id.values))  
        l = list(set(df_train_day_pre.user_id.values))     
        I = []
        C = []
        for posID in l:
            I.append(dic_i[posID])
        for posID in l:
            if posID not in dic_cov:
                C.append(0)
            else:
                C.append(dic_cov[posID])        
        print('开始平滑操作')           
        bs.update(I, C, 100000, 0.0001)
        print(bs.alpha, bs.beta) 
        
        
        print('构建平滑转化率')
        #dic_i_current = dict(Counter(df_train_day_current.item_id.values))
        #dic_cov_current = dict(Counter(df_train_day_current[df_train_day_current['is_trade'] == 1].item_id.values)) 
        dic_PH={}
        for pos in pos_all_list:
            if pos not in dic_i:
                dic_PH[pos] = (bs.alpha)/(bs.alpha + bs.beta)
            elif pos not in dic_cov:
                dic_PH[pos] = (bs.alpha)/(dic_i[pos] + bs.alpha + bs.beta)
            else:
                dic_PH[pos] = (dic_cov[pos] + bs.alpha)/(dic_i[pos] + bs.alpha+bs.beta)  
        #df_out=pd.DataFrame({'item_id':list(dic_PH.keys()),
         #                'PH_pos':list(dic_PH.values())})
    
        #df_out.to_csv('data/feature/PL_pos.csv',index=False)
        print('开始复制保存')
        ### !!! DAY19_data is data to merge
        DAY22_data = df_train_day_current[['instance_id','user_id','item_id']]
        user_ID_all = df_train_day_current['user_id']
        
        PH_user_id = []
  
        for i in user_ID_all:
             PH_user_id.append(dic_PH[i])
             
        DAY22_data['SmoothUserCVR'] = PH_user_id
        
        return DAY22_data
    
def Day23smooth_UserID():   
        df_train = readData()
        #pos_all_list=list(set(df_train.item_id.values))   
        df_train_day_pre = df_train[(df_train['day'] >= 18) & (df_train['day'] <= 22)]
        df_train_day_current = df_train[df_train['day'] == 23]
        pos_all_list=list(set(df_train_day_current.user_id.values)) 
        # -------------- day 20 -----------day 18 19 for alpha beta------------------------------
        #-------------------------------------
        print('开始统计pos平滑')
        #### calculate the alpha and beta with day18 data for day19
        bs = BayesianSmoothing(1.02853339139, 50.9925884934)  
        dic_i = dict(Counter(df_train_day_pre.user_id.values))
        dic_cov = dict(Counter(df_train_day_pre[df_train_day_pre['is_trade'] == 1].user_id.values))  
        l = list(set(df_train_day_pre.user_id.values))     
        I = []
        C = []
        for posID in l:
            I.append(dic_i[posID])
        for posID in l:
            if posID not in dic_cov:
                C.append(0)
            else:
                C.append(dic_cov[posID])        
        print('开始平滑操作')           
        bs.update(I, C, 100000, 0.0001)
        print(bs.alpha, bs.beta) 
        
        
        print('构建平滑转化率')
        #dic_i_current = dict(Counter(df_train_day_current.item_id.values))
        #dic_cov_current = dict(Counter(df_train_day_current[df_train_day_current['is_trade'] == 1].item_id.values)) 
        dic_PH={}
        for pos in pos_all_list:
            if pos not in dic_i:
                dic_PH[pos] = (bs.alpha)/(bs.alpha + bs.beta)
            elif pos not in dic_cov:
                dic_PH[pos] = (bs.alpha)/(dic_i[pos] + bs.alpha + bs.beta)
            else:
                dic_PH[pos] = (dic_cov[pos] + bs.alpha)/(dic_i[pos] + bs.alpha+bs.beta)  
        #df_out=pd.DataFrame({'item_id':list(dic_PH.keys()),
         #                'PH_pos':list(dic_PH.values())})
    
        #df_out.to_csv('data/feature/PL_pos.csv',index=False)
        print('开始复制保存')
        ### !!! DAY19_data is data to merge
        DAY23_data = df_train_day_current[['instance_id','user_id','item_id']]
        user_ID_all = df_train_day_current['user_id']
        
        PH_user_id = []
  
        for i in user_ID_all:
             PH_user_id.append(dic_PH[i])
             
        DAY23_data['SmoothUserCVR'] = PH_user_id
        
        return DAY23_data
    
def Day24smooth_UserID():   
        df_train = readData()
        #pos_all_list=list(set(df_train.item_id.values))   
        df_train_day_pre = df_train[(df_train['day'] >= 18) & (df_train['day'] <= 23)]
        df_train_day_current = df_train[df_train['day'] == 24]
        pos_all_list=list(set(df_train_day_current.user_id.values)) 
        # -------------- day 20 -----------day 18 19 for alpha beta------------------------------
        #-------------------------------------
        print('开始统计pos平滑')
        #### calculate the alpha and beta with day18 data for day19
        bs = BayesianSmoothing(1.04772468487 , 52.8343365855 )     
        dic_i = dict(Counter(df_train_day_pre.user_id.values))
        dic_cov = dict(Counter(df_train_day_pre[df_train_day_pre['is_trade'] == 1].user_id.values))  
        l = list(set(df_train_day_pre.user_id.values))     
        I = []
        C = []
        for posID in l:
            I.append(dic_i[posID])
        for posID in l:
            if posID not in dic_cov:
                C.append(0)
            else:
                C.append(dic_cov[posID])        
        print('开始平滑操作')           
        bs.update(I, C, 100000, 0.0001)
        print(bs.alpha, bs.beta) 
        
        
        print('构建平滑转化率')
        #dic_i_current = dict(Counter(df_train_day_current.item_id.values))
        #dic_cov_current = dict(Counter(df_train_day_current[df_train_day_current['is_trade'] == 1].item_id.values)) 
        dic_PH={}
        for pos in pos_all_list:
            if pos not in dic_i:
                dic_PH[pos] = (bs.alpha)/(bs.alpha + bs.beta)
            elif pos not in dic_cov:
                dic_PH[pos] = (bs.alpha)/(dic_i[pos] + bs.alpha + bs.beta)
            else:
                dic_PH[pos] = (dic_cov[pos] + bs.alpha)/(dic_i[pos] + bs.alpha+bs.beta)   
        #df_out=pd.DataFrame({'item_id':list(dic_PH.keys()),
         #                'PH_pos':list(dic_PH.values())})
    
        #df_out.to_csv('data/feature/PL_pos.csv',index=False)
        print('开始复制保存')
        ### !!! DAY19_data is data to merge
        DAY24_data = df_train_day_current[['instance_id','user_id','item_id']]
        user_ID_all = df_train_day_current['user_id']
        
        PH_user_id = []
  
        for i in user_ID_all:
             PH_user_id.append(dic_PH[i])
             
        DAY24_data['SmoothUserCVR'] = PH_user_id
        
        return DAY24_data
    
def Day25smooth_UserID():   
        df_train = readData()
        #pos_all_list=list(set(df_train.item_id.values))   
        df_train_day_pre = df_train[(df_train['day'] >= 18) & (df_train['day'] <= 24)]
        df_train_day_current = df_train[df_train['day'] == 25]
        pos_all_list=list(set(df_train_day_current.user_id.values)) 
        # -------------- day 20 -----------day 18 19 for alpha beta------------------------------
        #-------------------------------------
        print('开始统计pos平滑')
        #### calculate the alpha and beta with day18 data for day19
        bs = BayesianSmoothing(0.998830927426, 51.0217884086)  
        dic_i = dict(Counter(df_train_day_pre.user_id.values))
        dic_cov = dict(Counter(df_train_day_pre[df_train_day_pre['is_trade'] == 1].user_id.values))  
        l = list(set(df_train_day_pre.user_id.values))     
        I = []
        C = []
        for posID in l:
            I.append(dic_i[posID])
        for posID in l:
            if posID not in dic_cov:
                C.append(0)
            else:
                C.append(dic_cov[posID])        
        print('开始平滑操作')           
        bs.update(I, C, 100000, 0.0001)
        print(bs.alpha, bs.beta) 
        
        
        print('构建平滑转化率')
        #dic_i_current = dict(Counter(df_train_day_current.item_id.values))
        #dic_cov_current = dict(Counter(df_train_day_current[df_train_day_current['is_trade'] == 1].item_id.values)) 
        dic_PH={}
        for pos in pos_all_list:
            if pos not in dic_i:
                dic_PH[pos] = (bs.alpha)/(bs.alpha + bs.beta)
            elif pos not in dic_cov:
                dic_PH[pos] = (bs.alpha)/(dic_i[pos] + bs.alpha + bs.beta)
            else:
                dic_PH[pos] = (dic_cov[pos] + bs.alpha)/(dic_i[pos] + bs.alpha+bs.beta)   
        #df_out=pd.DataFrame({'item_id':list(dic_PH.keys()),
         #                'PH_pos':list(dic_PH.values())})
    
        #df_out.to_csv('data/feature/PL_pos.csv',index=False)
        print('开始复制保存')
        ### !!! DAY19_data is data to merge
        DAY25_data = df_train_day_current[['instance_id','user_id','item_id']]
        user_ID_all = df_train_day_current['user_id']
        
        PH_user_id = []
  
        for i in user_ID_all:
             PH_user_id.append(dic_PH[i])
             
        DAY25_data['SmoothUserCVR'] = PH_user_id
        
        return DAY25_data

def concatUserDayCVR():
   data = readData()
   #DAY18_data = Day18smooth_ItemID()
   DAY19_data = Day19smooth_UserID()
   DAY20_data = Day20smooth_UserID()
   DAY21_data = Day21smooth_UserID()
   DAY22_data = Day22smooth_UserID()
   DAY23_data = Day23smooth_UserID()
   DAY24_data = Day24smooth_UserID()
   DAY25_data = Day25smooth_UserID()
   frames = [DAY19_data, DAY20_data, DAY21_data, DAY22_data, DAY23_data, DAY24_data, DAY25_data]
   result = pd.concat(frames)
   resultAppend = pd.merge(data, result, on=['instance_id','item_id','user_id'], how = 'left')
   userCVR = resultAppend['SmoothUserCVR']
   userCVR = userCVR.fillna(0)
   np.save('./datasets/UserBayesPH.npy', userCVR)
   return userCVR
'''
def smooth_ItembrandID():
        #return mergeData
        test = pd.read_csv('../data/round1_ijcai_18_test_a_20180301.txt', sep = ' ')   
        train = pd.read_csv('../data/round1_ijcai_18_train_20180301.txt', sep = ' ')
        train.drop_duplicates(inplace = True)
        key = list( test )
        mergeData = pd.concat([train, test], keys = key)
        mergeData = mergeData.reset_index( drop = True )
        pos_all_list=list(set(mergeData.item_brand_id.values)) 
        #del df_pos    
        print('载入完成，开始拼接')    
        df_train=pd.read_csv('../data/round1_ijcai_18_train_20180301.txt', sep = ' ')
        
        #-------------------------------------
        print('开始统计pos平滑')
        bs = BayesianSmoothing(2.06458560045, 104.94182907)    
        dic_i=dict(Counter(df_train.item_brand_id.values))
        dic_cov=dict(Counter(df_train[df_train['is_trade'] == 1].item_brand_id.values))  
        l=list(set(df_train.item_brand_id.values))     
        I=[]
        C=[]
        for posID in l:
            I.append(dic_i[posID])
        for posID in l:
            if posID not in dic_cov:
                C.append(0)
            else:
                C.append(dic_cov[posID])        
        print('开始平滑操作')           
        bs.update(I, C, 1, 0.0000000001)
        print(bs.alpha, bs.beta)  
        print('构建平滑转化率')
        dic_PH={}
        for pos in pos_all_list:
            if pos not in dic_i:
                dic_PH[pos]=(bs.alpha)/(bs.alpha + bs.beta)
            elif pos not in dic_cov:
                dic_PH[pos]=(bs.alpha)/(dic_i[pos] + bs.alpha + bs.beta)
            else:
                dic_PH[pos]=(dic_cov[pos] + bs.alpha)/(dic_i[pos] + bs.alpha + bs.beta)   
        #df_out=pd.DataFrame({'item_id':list(dic_PH.keys()),
         #                'PH_pos':list(dic_PH.values())})
    
#df_out.to_csv('data/feature/PL_pos.csv',index=False)
        print('开始复制保存')

        item_brand_ID_all = mergeData['item_brand_id']

        PH_Item_ID = []
        #merge_Item_ID = mergeData['IT']
        for i in item_brand_ID_all:
             PH_Item_ID.append(dic_PH[i])
        #mergeData['PH_item_ID'] = PH_Item_ID
        return PH_Item_ID
'''    
if __name__ == "__main__":
   #data = readData()
   result = concatUserDayCVR()
   #DAY19_data = Day19smooth_UserID()
   #DAY20_data = Day20smooth_UserID()
   #DAY21_data = Day21smooth_UserID()
   #DAY22_data = Day22smooth_UserID()
   #DAY23_data = Day23smooth_UserID()
   #DAY24_data = Day24smooth_UserID()
   #DAY25_data = Day25smooth_UserID()

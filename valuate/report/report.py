import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.externals import joblib
from valuate.db import process_tables
from valuate.predict.predict_batch import Predict
from valuate.conf import global_settings as gl


def generate_days(df):
    """
    计算使用时间(通用)
    """
    online_time = str(df['year'])+'-'+str(df['month'])+'-'+'1'
    online_time = datetime.strptime(online_time, '%Y-%m-%d')
    if str(df['sold_time']) == 'nan':
        transaction_time = datetime.strptime(str(df['expired_at']), "%Y-%m-%d %H:%M:%S")
        return (transaction_time - online_time).days
    else:
        transaction_time = datetime.strptime(str(df['sold_time']), "%Y-%m-%d %H:%M:%S")
        return (transaction_time - online_time).days


def generate_deal_days(df):
    """
    计算成交表使用时间
    """
    return (datetime.strptime(df['deal_date'], '%Y-%m-%d') - datetime.strptime(df['reg_date'], '%Y-%m-%d')).days


def query_competed_tables():
    process_tables.store_competed_tables()


class Report(object):

    def __init__(self):
        """
        初始化
        """
        # 查询竞品表
        self.car_source = {}
        self.final = {}
        self.ret = {}

    def process_competed_tables(self):
        """
        竞品表的相关处理
        """
        # 加载竞品表
        self.final = pd.read_csv('../tmp/report/eval_source.csv')
        self.final = self.final[self.final['source_type'] != '0']
        self.final.reset_index(inplace=True)
        self.final = self.final.sort_values(by=['car_id', 'domain'])
        self.final = self.final.drop(['id', 'created', 'title', 'pub_time'], axis=1)

        # 对交易类型进行分类
        self.final['category'] = np.NaN
        self.final.loc[(self.final['source_type'].isin(gl.C_2_C)), 'category'] = 'c_2_c'
        self.final.loc[(self.final['source_type'].isin(gl.B_2_C)), 'category'] = 'b_2_c'

        # 计算使用时间
        self.final['use_time'] = self.final.apply(generate_days, axis=1)

        # 重置索引
        self.final = self.final.drop(['index', 'expired_at', 'year', 'month'], axis=1)
        self.final = self.final.rename(columns={'sold_time': 'deal_date'})
        self.final.reset_index(inplace=True)
        self.final = self.final.drop('index', axis=1)

    def process_deal_tables(self):
        """
        成交表相关处理
        """
        deal_records = pd.read_csv('../tmp/report/deal_records.csv')
        eval_deal_source = pd.read_csv('../tmp/report/eval_deal_source.csv')
        open_model_detail = pd.read_csv('../tmp/train/open_model_detail.csv')
        model_detail_map = pd.read_csv('predict/map/model_detail_map.csv')
        model_detail_map = model_detail_map.loc[:, ['model_slug', 'model_detail_slug']]
        deal_records = deal_records.loc[:, ['id', 'model_slug', 'model_detail_slug', 'city', 'mile', 'price', 'deal_date', 'reg_date','deal_type']]
        eval_deal_source = eval_deal_source.loc[:, ['deal_id', 'good', 'domain', 'status']]
        open_model_detail = open_model_detail.loc[:, ['price_bn', 'model_detail_slug']]

        deal_records = deal_records.rename(columns={'id': 'car_id'})
        eval_deal_source = eval_deal_source.rename(columns={'deal_id': 'car_id'})
        final = eval_deal_source.merge(deal_records, how='left', on='car_id')
        final['mile'] = final['mile'] / 10000
        final['price'] = final['price'] / 100

        final = final.drop(final[final['deal_type'].isin([4])].index)
        final['use_time'] = final.apply(generate_deal_days, axis=1)
        final['source_type'] = final['deal_type'].map(gl.DEAL_TYPE_MAP_SOURCE_TYPE)
        final['category'] = final['deal_type'].map(gl.DEAL_TYPE_MAP_CATEGORY)
        final = final.drop(['reg_date', 'deal_type'], axis=1)
        final = final.merge(open_model_detail, how='left', on='model_detail_slug')

        self.final = self.final.append(final, ignore_index=True)
        self.final = self.final.drop(self.final[(self.final['category'] == 'c_2_c') & (self.final['domain'] == 'jingzhengu.com')].index)
        self.final = self.final[self.final['status'] == '1']
        # 将odealer,cpo,personal类型修改为dealer和cpersonal
        self.final.loc[(self.final['source_type'].isin(['odealer', 'cpo'])), 'source_type'] = 'dealer'
        self.final.loc[(self.final['source_type'].isin(['personal'])), 'source_type'] = 'cpersonal'
        self.final.reset_index(inplace=True)
        self.final = self.final.drop(['index', 'model_slug'], axis=1)
        self.final = self.final.merge(model_detail_map, how='left', on='model_detail_slug')

    def predict_condition(self):
        """
        预测车况
        """
        # car_condtion = self.final.loc[:, ['use_time', 'mile']]
        # # car_condtion['transfer_owner'] = 0
        # car_condition = self.car_condition_model.predict(car_condtion)
        # self.final['car_condition'] = pd.Series(car_condition).values
        # self.final = self.final.merge(self.car_condition_match, how='left', on='car_condition')
        # self.final['desc'] = self.final['desc'].str.lower()
        # self.final = self.final.rename(columns={'desc': 'condition'})
        # self.final = self.final.drop(['source_type', 'origin_price'], axis=1)
        # # 删除车况为bad的记录
        # self.final = self.final.drop(self.final[self.final['condition'] == 'bad'].index)
        # # 重置索引
        # self.final.reset_index(inplace=True)
        # self.final = self.final.drop('index', axis=1)

    def find_predict_price(self):
        """
        查找预测值
        """
        def process_category_price(df):
            if df['exist'] == 'Y':
                if df['category'] == 'c_2_c':
                    if df['domain'] == 'jingzhengu.com':
                        return 0
                    elif df['domain'] == 'gongpingjia.com':
                        return df['good'].split(' ')[1]
                    elif df['domain'] == 'che300.com':
                        return df['good'].split(',')[3]
                elif df['category'] == 'b_2_c':
                    if df['domain'] == 'jingzhengu.com':
                        return df['good'].split(',')[2]
                    elif df['domain'] == 'gongpingjia.com':
                        return df['good'].split(' ')[2]
                    elif df['domain'] == 'che300.com':
                        return df['good'].split(',')[5]
                elif df['category'] == 'c_2_b':
                    if df['domain'] == 'jingzhengu.com':
                        return df['good'].split(';')[0].split(',')[1]
                    elif df['domain'] == 'gongpingjia.com':
                        return df['good'].split(' ')[0]
                    elif df['domain'] == 'che300.com':
                        return df['good'].split(',')[1]

        self.final['exist'] = 'Y'
        self.final.loc[(self.final['good'].isnull()), 'exist'] = 'N'
        self.final.loc[(self.final['good'] == '0,0'), 'exist'] = 'N'
        self.final['predict_price'] = self.final.apply(process_category_price, axis=1)

    def keep_all_have_data(self):
        """
        只保留竞品都有的数据
        """
        standard_ids = []
        car_ids = set(self.final.car_id.values)

        # 保留三家平台都有的车源数据
        temp = self.final[self.final['category'] == 'c_2_c']
        car_ids = list(set(temp.car_id.values))
        for car_id in car_ids:
            if len(temp.loc[(temp['car_id'] == car_id), :]) == 2:
                standard_ids.append(car_id)

        temp = self.final[self.final['category'] == 'b_2_c']
        car_ids = list(set(temp.car_id.values))
        for car_id in car_ids:
            if len(temp.loc[(temp['car_id'] == car_id), :]) == 3:
                standard_ids.append(car_id)

        temp = self.final[self.final['category'] == 'c_2_b']
        car_ids = list(set(temp.car_id.values))
        for car_id in car_ids:
            if len(temp.loc[(temp['car_id'] == car_id), :]) == 3:
                standard_ids.append(car_id)

        self.final = self.final[self.final['car_id'].isin(standard_ids)]
        # 重置索引
        self.final.reset_index(inplace=True)
        self.final = self.final.drop('index', axis=1)
        return self.final

    def generate_temp_csv_with_predict(self):
        """
        生成jupyter使用的临时csv,带本地预测
        """
        # 查询竞争数据
        # query_competed_tables()
        # 处理竞品表
        self.process_competed_tables()
        # 处理成交表
        self.process_deal_tables()
        # 查找预测值
        self.find_predict_price()
        # 删除没有预测值的记录
        self.final = self.final.drop(self.final[self.final['exist'] == 'N'].index)
        # 重置索引
        self.final.reset_index(inplace=True)
        self.final = self.final.drop('index', axis=1)

        # 使用最新的模型预测
        gongpingjia = self.final.loc[(self.final['domain'] == 'gongpingjia.com'), :]
        predict = Predict()
        result = predict.predict_new(gongpingjia.loc[:, ['car_id', 'city', 'model_slug', 'model_detail_slug',  'source_type', 'price_bn', 'use_time', 'mile']])
        result = result.loc[:, ['car_id', 'predict_price']]
        gongpingjia = gongpingjia.drop('predict_price', axis=1)
        gongpingjia = gongpingjia.merge(result, how='left', on='car_id')
        gongpingjia['predict_price'] = gongpingjia['predict_price'] / 100
        # gongpingjia['predict_price'] = gongpingjia['predict_price'].astype(int)
        self.final = self.final.drop(self.final[self.final['domain'] == 'gongpingjia.com'].index)
        self.final = self.final.append(gongpingjia, ignore_index=True)
        self.final = self.final.sort_values(by=['car_id', 'domain'])

        # 保留各平台都有的数据并存储
        self.keep_all_have_data()
        self.final.to_csv('../tmp/report/report_compete_data.csv', index=False)


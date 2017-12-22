import os
import json
import time
import pandas as pd
from sqlalchemy import create_engine
from valuate.db import db_operate
from valuate.conf import global_settings as gl

from valuate.exception.api_error import ApiParamsValueError
from valuate.exception.api_error import ApiParamsTypeError

# 重定位根路径
path = os.path.abspath(os.path.dirname(gl.__file__))
path = path.replace('conf', '')

# 加载省份城市匹配表
province_city_map = pd.read_csv(path+'predict/map/province_city_map.csv')
cities = list(set(province_city_map.city.values))
province_city_map = province_city_map.set_index('city')

# 加载省、车型，流行度匹配表
province_popularity_map = pd.read_csv(path+'predict/map/province_popularity_map.csv')
province_popularity_map['encode'] = province_popularity_map['model_slug']+'_'+province_popularity_map['province']
province_popularity_index = list(set(province_popularity_map.encode.values))
province_popularity_map = province_popularity_map.set_index('encode')

# 加载车型,款系映射表
model_detail_map = pd.read_csv(path+'predict/map/model_detail_map.csv')
models = list(set(model_detail_map.model_detail_slug.values))
model_detail_map = model_detail_map.set_index('model_detail_slug')

# 返回结构格式
result_map = pd.DataFrame(columns=['intent', 'intent_source', 'predict_price'])
result_map['intent'] = pd.Series(gl.INTENT_TYPE)
result_map['intent_source'] = pd.Series(gl.INTENT_TYPE_CAN)

# 调整值映射表
adjust_profit = pd.read_csv('predict/map/adjust_profit_map.csv')
adjust_profit['encode'] = adjust_profit['model_detail_slug']+'_'+adjust_profit['popularity']
model_detail_slug_popularity_index = list(set(adjust_profit.encode.values))
adjust_profit = adjust_profit.set_index('encode')

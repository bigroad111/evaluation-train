import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import time

from sklearn.externals import joblib
from valuate.conf import global_settings as gl

from valuate.exception.api_error import ApiParamsValueError
from valuate.exception.api_error import ApiParamsTypeError


# 重定位根路径
path = os.path.abspath(os.path.dirname(gl.__file__))
path = path.replace('conf', '')

# 加载省份城市匹配表
province_city_map = pd.read_csv(path+'predict/map/province_city_map.csv')

# 加载省、车型，流行度匹配表
province_popularity_map = pd.read_csv(path+'predict/map/province_popularity_map.csv')
province_popularity_map['index'] = province_popularity_map['province']+province_popularity_map['model_slug']
province_popularity_map = province_popularity_map.set_index('index')

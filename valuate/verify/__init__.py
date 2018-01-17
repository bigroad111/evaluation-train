import os
import time

import pandas as pd
import multiprocessing
from valuate.conf import global_settings as gl

# 重定位根路径
path = os.path.abspath(os.path.dirname(gl.__file__))
path = path.replace('conf', '')

# 加载省份城市匹配表
province_city_map = pd.read_csv(path+'predict/map/province_city_map.csv')
cities = list(set(province_city_map.city.values))

# 加载车型,款系映射表
model_detail_map = pd.read_csv(path+'predict/map/model_detail_map.csv')
models = list(set(model_detail_map.model_slug.values))
details = list(set(model_detail_map.model_detail_slug.values))


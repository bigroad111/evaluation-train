# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '../')
import time
import os
import pandas as pd
# pd.set_option('precision', 2)
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

from valuate.db import db_operate,process_tables
from valuate.common import common_func as cf
from valuate.report.report import Report
from valuate.predict.predict import Predict as pred
from valuate.predict.predict_api import Predict as api
from valuate.predict.predict_batch import Predict as batch
from valuate.train.feature_engineering import FeatureEngineering
from valuate.train.stacking import Stacking
from valuate.verify.verify import Verify
from valuate.train.manual import Manual
from valuate.conf import global_settings as gl
from valuate.exception.api_error import ApiParamsValueError
from valuate.exception.api_error import ApiParamsTypeError

import cProfile

if __name__ == "__main__":
    # 使用源数据进行特征工程
    # fe = FeatureEngineering()
    # fe.execute()

    # 加载特征工程处理后的训练数据
    # valuated_model = pd.read_csv('predict/map/valuated_model_detail.csv')
    # models = list(set(valuated_model.model_slug.values))
    #
    # # 训练stacking模型
    # stack = Stacking()
    # stack.execute_multitude(models)
    # # stack.grid_search()
    # 后续人工处理
    # manual = Manual()
    # manual.execute()

    # valuated_model = pd.read_csv('predict/map/valuated_model_detail.csv')
    # valuated_model.to_csv('predict/map/need_valuated_model.csv', index=False)
    # predict = pred()
    # predict.predict_all_test_data()
    # predict.process_all_result_data()
    # predict.combine_all_result_data()

    # pr = cProfile.Profile()
    # pr.enable()
    # while True:
    # time1 = time.time()
    # predict = api()
    # result = predict.predict(city='重庆', model_detail_slug='13419_ah', use_time=51, mile=6.9, ret_type='normal')
    # print(result)
    # result = predict.future_price_trend(city='广州', model_detail_slug='13648_ah', use_time=62, mile=13, ret_type='normal')
    # print(result)
    # result = predict.history_price_trend(city='广州', model_detail_slug='13648_ah', use_time=62, mile=13, ret_type='normal')
    # time2 = time.time()
    # print(time2 - time1)
    # # # # pr.disable()
    # # # # pr.print_stats(sort="calls")
    # print(result)

    # 单元测试
    verify = Verify()
    # verify.generate_verify_tables()
    # verify.execute_verify()
    result = verify.load_valuated_model_details()
    print(result)

    # # 竞品分析报告生成
    # report = Report()
    # report.generate_temp_csv_with_predict()

    # 因训练完新模型需要更新的相关表
    # process_tables.store_need_udpate_tables()
    # process_tables.process_need_udpate_tables()

    # print(gl.PRODUCE_VALUATE_ENGINE_INNER)

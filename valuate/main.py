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

    # time1 = time.time()
    # result = predict.predict_single(model_slug='i-benz-e', model_detail_slug='18051_ah', city='深圳', use_time=40)
    # time2 = time.time()
    # print(time2 - time1)
    # print(result)

    # pr = cProfile.Profile()
    # pr.enable()
    # while True:
    time1 = time.time()
    predict = api()
    result = predict.predict(city='南昌', model_detail_slug='109356_autotis', use_time=240, mile=1.9, ret_type='normal')
    # result = predict.future_price_trend(city='北京', model_detail_slug='109356_autotis', use_time=229, mile=-5, ret_type='normal')
    # result = predict.history_price_trend(city='贺州', model_detail_slug='19429_ah', use_time=10, mile=0, ret_type='normal')
    time2 = time.time()
    print(time2 - time1)
    # # # pr.disable()
    # # # pr.print_stats(sort="calls")
    print(result)

    # 单元测试
    # verify = Verify()
    # verify.generate_verify_tables()
    # verify.execute_verify()
    # result = verify.load_valuated_model_details()
    # print(result)

    # # 竞品分析报告生成
    # report = Report()
    # report.generate_temp_csv_with_predict()

    # 因训练完新模型需要更新的相关表
    # process_tables.store_need_udpate_tables()
    # process_tables.process_need_udpate_tables()
    # process_tables.update_tables_to_local_db()

    # print(gl.PRODUCE_VALUATE_ENGINE_INNER)

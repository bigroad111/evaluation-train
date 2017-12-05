from valuate.predict import *


def get_profit_rate(intent, popularity):
    """
    获取畅销系数
    """
    # 按畅销程度分级,各交易方式相比于标价的固定比例
    profits = gl.PROFITS
    profit = profits[popularity]
    # 计算各交易方式的价格相比于标价的固定比例
    if intent == 'sell':
        # 商家收购价相比加权平均价的比例
        profit_rate = 1 - profit[0] - profit[1]
    elif intent == 'buy':
        # 商家真实售价相比加权平均价的比例
        profit_rate = 1 - profit[0]
    elif intent == 'release':
        # 建议标价相比加权平均价的比例
        profit_rate = 1
    elif intent == 'private':
        # C2C价格相比加权平均价的比例
        profit_rate = 1 - profit[0] - profit[2]
    elif intent == 'lowest':
        # 最低成交价相比加权平均价的比例
        profit_rate = 1 - profit[0] - profit[1] - profit[3]
    elif intent == 'cpo':
        # 认证二手车价相比加权平均价的差异比例
        profit_rate = 1 - profit[0] - profit[8]
    elif intent == 'replace':
        # 4S店置换价相比加权平均价的比例
        profit_rate = 1 - profit[0] - profit[4]
    elif intent == 'auction':
        # 拍卖价相比加权平均价的差异比例
        profit_rate = 1 - profit[0] - profit[5]
    elif intent == 'avg-buy':
        # 平均买车价相比加权平均价的差异比例
        profit_rate = 1 - profit[0] - profit[7]
    elif intent == 'avg-sell':
        # 平均卖车价价相比加权平均价的差异比例
        profit_rate = 1 - profit[0] - profit[6]
    return profit_rate


def get_all_condition_values(valuate_price, car_condition):
    """
    根据标价计算4个级别车况的价
    """
    values = []
    for con in gl.CAR_CONDITION_VALUES:
        values.append(valuate_price - (gl.CAR_CONDITION_COEFFICIENT[car_condition] - gl.CAR_CONDITION_COEFFICIENT[con]) * valuate_price)
    return values


def cal_intent_condition(valuate_price, popularity):
    """
    计算所有交易方式的4个级别车况价
    """
    conditions = get_all_condition_values(valuate_price, 'good')
    profit_rates = []
    for i, con in enumerate(gl.INTENT_TYPE):
        profit_rate = get_profit_rate(con, popularity)
        profit_rates.append(profit_rate)
    df1 = pd.DataFrame(profit_rates)
    df2 = pd.DataFrame([conditions])
    all_map = df1.dot(df2)
    all_map.columns = ['excellent', 'good', 'fair', 'bad']
    all_map['intent'] = pd.Series(gl.INTENT_TYPE).values
    all_map = all_map.loc[:, ['intent', 'excellent', 'good', 'fair', 'bad']]
    all_map[['excellent', 'good', 'fair', 'bad']] = all_map[['excellent', 'good', 'fair', 'bad']].astype(int)
    return all_map


def cal_intent_condition_method2(prices,price_bn):
    """
    计算所有交易方式的4个级别车况价
    """
    if(prices[2] * 1.03) > price_bn:
        rate = (prices[2] * 1.03) / price_bn
        prices = prices / rate

    df1 = pd.DataFrame(prices)
    df2 = pd.DataFrame([gl.CAR_CONDITION_COEFFICIENT_VALUES])
    all_map = df1.dot(df2)
    all_map.columns = ['excellent', 'good', 'fair', 'bad']
    all_map['intent'] = pd.Series(gl.INTENT_TYPE).values
    all_map = all_map.loc[:, ['intent', 'excellent', 'good', 'fair', 'bad']]
    all_map[['excellent', 'good', 'fair', 'bad']] = all_map[['excellent', 'good', 'fair', 'bad']].astype(int)
    return all_map


def process_index(df):
    """
    装载预测表的索引
    """
    return df['model_detail_slug'] + '_' + str(df['use_time'])


def process_mile(df):
    """
    mile处理
    """
    if df['mile_per_month'] < gl.MILE_THRESHOLD:
        return df['predict_price'] + (gl.MILE_RATE/3) * df['predict_price'] * (1 - df['mile_per_month']/gl.MILE_THRESHOLD)
    else:
        return df['predict_price'] - (gl.MILE_RATE * 2/3) * df['predict_price'] * (1 - gl.MILE_THRESHOLD/df['mile_per_month'])


def process_profit_rate(df):
    """
    畅销系数处理
    """
    return get_profit_rate(df['intent'], df['popularity'])


def process_buy_profit_rate(df):
    """
    畅销系数处理
    """
    return get_profit_rate(df['intent_source'], df['popularity'])


def process_unreasonable_history_price(data, nums):
    """
    处理不合理历史价格趋势
    """
    if nums == 0:
        return data

    temp = data[1:]
    temp.sort()
    for i, value in enumerate(temp):
        data[i+1] = temp[i]

    for i in range(0, nums):
        if data[i] >= data[i + 1]:
            data[i + 1] = int(data[i] * 1.0083)

    return data


def process_unreasonable_future_price(data):
    """
    处理不合理未来价格趋势
    """
    temp = data[1:]
    temp.sort(reverse=True)
    for i, value in enumerate(temp):
        data[i+1] = temp[i]

    for i in range(0, 3):
        if data[i] <= data[i + 1]:
            data[i + 1] = int(data[i] * 0.9)

    return data


class Predict(object):

    def __init__(self):
        """
        加载各类匹配表和模型
        """
        self.test_level1 = []
        self.test_level2 = []
        self.predict_hedge = []
        self.predict_price = []

    def map_feature_encode(self, car_source):
        """
        1.匹配特征编码
        """
        self.test_level1 = car_source

    def model_predict(self):
        """
        2.模型预测
        """
        # 估值预测
        pred = self.test_level1.loc[:, feature_name]
        pred['city_encode'] = 300
        self.predict_hedge = np.exp(valuate_xgb.predict(xgb.DMatrix(pred)))
        # self.predict_hedge = valuate_xgb.predict(xgb.DMatrix(pred))
        self.test_level1['predict_hedge'] = pd.Series(self.predict_hedge).values
        print(self.test_level1['predict_hedge'].values)
        # print(self.test_level1)

    # def add_process_mile(self):
    #     """
    #     3.根据公里数修正预测值
    #     """
    #     self.test_level1['mile_per_month'] = self.test_level1['mile'] / self.test_level1['use_time']
    #     self.test_level1['predict_price'] = self.test_level1.apply(process_mile, axis=1)
    #
    # def add_process_condition(self):
    #     """
    #     4.预测车况
    #     """
    #     # 评估当前车况
    #     # temp = self.test_level1.loc[:, ['use_time', 'mile']]
    #     # car_condition = self.car_condition_model.predict(temp)
    #     # self.test_level1['car_condition'] = pd.Series(car_condition).values
    #     # self.test_level1 = self.test_level1.merge(self.car_condition_match, how='left', on='car_condition')
    #     # self.test_level1['desc'] = self.test_level1['desc'].str.lower()
    #     # self.test_level1 = self.test_level1.rename(columns={'desc': 'condition'})
    #
    # def add_process_prices_relate(self):
    #     """
    #     4.人工处理预测不合理的三类价格
    #     """
    #     private = self.test_level1.loc[(self.test_level1['intent'] == 'private'), 'predict_price'].values[0]
    #     buy = self.test_level1.loc[(self.test_level1['intent'] == 'buy'), 'predict_price'].values[0]
    #     sell = self.test_level1.loc[(self.test_level1['intent'] == 'sell'), 'predict_price'].values[0]
    #     # 计算三类价格比例关系
    #     private_buy_rate = (buy - private) / private
    #     # 人工处理预测不合理的三类价格
    #     if (private_buy_rate < 0) | (abs(private_buy_rate) > 0.12) | (abs(private_buy_rate) <= 0.055):
    #         private = int(buy * (1 - 0.0875))
    #         self.test_level1.loc[(self.test_level1['intent'] == 'private'), 'predict_price'] = int(buy * (1 - 0.0875))
    #
    #     private_sell_rate = (sell - private) / private
    #     if (private_sell_rate > 0) | (abs(private_sell_rate) > 0.085) | (abs(private_sell_rate) <= 0.02):
    #         self.test_level1.loc[(self.test_level1['intent'] == 'sell'), 'predict_price'] = int(private * (1 - 0.0525))
    #
    #     # 计算所有交易类型
    #     self.test_level1 = cal_intent_condition_method2(self.test_level1.predict_price.values, self.test_level1.loc[0, 'price_bn'])
    #     # if ~(0.055 <= private_buy_rate <= 0.12):
    #     #     self.test_level1.loc[(self.test_level1['source_type'] == 'dealer'), 'predict_price'] = int(private * 1.0875)
    #     #
    #     # if ~(-0.085 <= private_sell_rate <= -0.02):
    #     #     self.test_level1.loc[(self.test_level1['intent'] == 'sell'), 'predict_price'] = int(private * (1 - 0.0525))
    #
    # def add_process_intent(self, model_detail_slug):
    #     """
    #     5.根据交易方式修正预测值
    #     """
    #     # self.test_level1['popularity'] = self.test_level1['popularity'].fillna('C')
    #     self.test_level1['profit_rate'] = self.test_level1.apply(process_profit_rate, axis=1)
    #     self.test_level1['buy_profit_rate'] = self.test_level1.apply(process_buy_profit_rate, axis=1)
    #     self.test_level1['predict_price'] = self.test_level1['predict_price'] / self.test_level1['buy_profit_rate']
    #     self.test_level1['predict_price'] = self.test_level1['profit_rate'] * self.test_level1['predict_price']
    #
    # def check_params_value(self, city, model_detail_slug, use_time, mile, category):
    #     # 校验city
    #     if city not in cities:
    #         raise ApiParamsValueError('city', city, 'Unknown city!')
    #     # 校验model
    #     if model_detail_slug not in models:
    #         raise ApiParamsValueError('model_detail_slug', model_detail_slug, 'Unknown model!')
    #     # 校验mile
    #     if not ((isinstance(mile, int)) | (isinstance(mile, float))):
    #         raise ApiParamsTypeError('mile', mile, 'Mile must be int or float!')
    #     elif mile < 0:
    #         raise ApiParamsValueError('mile', mile, 'Mile must be greater than zero!')
    #     # 校验use_time
    #     if not isinstance(use_time, int):
    #         raise ApiParamsTypeError('use_time', use_time, 'Use_time must be int!')
    #     if category == 'valuate':
    #         if (use_time < 1) | (use_time > 7500):
    #             raise ApiParamsValueError('use_time', use_time, 'The use_time of Forecast must be in 1-7500!')
    #     elif category == 'history':
    #         if (use_time < 1) | (use_time > 7500):
    #             raise ApiParamsValueError('use_time', use_time, 'The use_time of historical trend must be in 1-7500!')
    #     elif category == 'future':
    #         if (use_time < 1) | (use_time > 7500):
    #             raise ApiParamsValueError('use_time', use_time, 'The use_time of future trend must be in 1-7500!')

    def predict_new(self, city='深圳', model_detail_slug='model_25023_cs', use_time=12, mile=2):
        """
        预测返回
        """
        # 预测模型能够预测的所有类型
        car_source = pd.DataFrame()
        car_source['source_type'] = pd.Series(gl.INTENT_TYPE_CAN)
        # 获取交易类型编码
        car_source['source_type_encode'] = pd.Series(source_type_encode)
        # 获取品牌和车型编码
        car_source['model_detail_slug'] = model_detail_slug
        car_source = car_source.merge(encode_model_detail_map, how='left', on='model_detail_slug')
        car_source = car_source.merge(encode_model_detail_slug, how='left', on='model_detail_slug')
        car_source['city'] = city
        car_source = car_source.merge(encode_city, how='left', on='city')
        # 获取畅销度编码,根据城市获取省份
        model_slug = car_source.loc[0, 'model_slug']
        province = province_city_map.loc[city, 'province']
        if str(province+model_slug) in province_popularity_map.index:
            car_source['popularity'] = province_popularity_map.loc[str(province+model_slug), 'popularity']
        else:
            car_source['popularity'] = 'C'
        car_source = car_source.merge(encode_popularity, how='left', on='popularity')
        # 计算其他参数
        car_source['use_time'] = use_time
        car_source['mile'] = mile
        car_source['mile_per_month'] = mile / use_time
        car_source['price_bn'] = car_source['price_bn'] * 10000

        self.map_feature_encode(car_source)
        self.model_predict()
        return self.test_level1

    # def predict(self, city='深圳', model_detail_slug='model_25023_cs', use_time=365, mile=2):
    #     """
    #     预测返回
    #     """
    #     # 预测模型能够预测的所有类型
    #     price_bn = model_detail_map.loc[model_detail_slug, 'price_bn']
    #     car_source = pd.DataFrame()
    #     car_source['intent'] = pd.Series(gl.INTENT_TYPE)
    #     car_source['source_type'] = pd.Series(gl.INTENT_TYPE_CAN)
    #     # 获取交易类型编码
    #     car_source['source_type_encode'] = pd.Series(source_type_encode)
    #     # 获取城市编码
    #     car_source['city'] = city
    #     car_source['city_encode'] = encode_city.loc[city, 'city_encode']
    #     # 获取款型编码
    #     car_source['model_detail_slug'] = model_detail_slug
    #     car_source['model_detail_slug_encode'] = encode_model_detail_slug.loc[model_detail_slug, 'model_detail_slug_encode']
    #     car_source['use_time'] = use_time
    #     car_source['mile'] = mile
    #     car_source['price_bn'] = price_bn * 10000
    #     car_source['intent_source'] = car_source['source_type'].map(gl.INTENT_MAP)
    #
    #     # 根据款型获取车型
    #     model_slug = model_detail_map.loc[model_detail_slug, 'model_slug']
    #     car_source['model_slug'] = model_slug
    #     # 根据城市获取省份
    #     province = province_city_map.loc[city, 'province']
    #     car_source['province'] = province
    #     # 根据城市,车型获取流行度
    #     if str(province+model_slug) in province_popularity_map.index:
    #         car_source['popularity'] = province_popularity_map.loc[str(province+model_slug), 'popularity']
    #     else:
    #         car_source['popularity'] = 'C'
    #
    #     self.map_feature_encode(car_source)
    #     self.model_predict()
    #     self.add_process_mile()
    #     self.add_process_intent(model_detail_slug)
    #     self.add_process_prices_relate()
    #     return self.test_level1
    #
    # def predict_to_dict(self, city='深圳', model_detail_slug='model_25023_cs', use_time=365, mile=2):
    #     """
    #     预测返回
    #     """
    #     # 校验参数
    #     self.check_params_value(city, model_detail_slug, use_time, mile, category='valuate')
    #     # 预测模型能够预测的所有类型
    #     price_bn = model_detail_map.loc[model_detail_slug, 'price_bn']
    #     car_source = pd.DataFrame()
    #     car_source['intent'] = pd.Series(gl.INTENT_TYPE)
    #     car_source['source_type'] = pd.Series(gl.INTENT_TYPE_CAN)
    #     # 获取交易类型编码
    #     car_source['source_type_encode'] = pd.Series(source_type_encode)
    #     # 获取城市编码
    #     car_source['city'] = city
    #     car_source['city_encode'] = encode_city.loc[city, 'city_encode']
    #     # 获取款型编码
    #     car_source['model_detail_slug'] = model_detail_slug
    #     car_source['model_detail_slug_encode'] = encode_model_detail_slug.loc[model_detail_slug, 'model_detail_slug_encode']
    #     car_source['use_time'] = use_time
    #     car_source['mile'] = mile
    #     car_source['price_bn'] = price_bn * 10000
    #     car_source['intent_source'] = car_source['source_type'].map(gl.INTENT_MAP)
    #
    #     # 根据款型获取车型
    #     model_slug = model_detail_map.loc[model_detail_slug, 'model_slug']
    #     car_source['model_slug'] = model_slug
    #     # 根据城市获取省份
    #     province = province_city_map.loc[city, 'province']
    #     car_source['province'] = province
    #     # 根据城市,车型获取流行度
    #     if str(province+model_slug) in province_popularity_map.index:
    #         car_source['popularity'] = province_popularity_map.loc[str(province+model_slug), 'popularity']
    #     else:
    #         car_source['popularity'] = 'C'
    #
    #     self.map_feature_encode(car_source)
    #     self.model_predict()
    #     self.add_process_mile()
    #     self.add_process_intent(model_detail_slug)
    #     self.add_process_prices_relate()
    #     return self.test_level1.to_dict('records')
    #
    # def history_price_trend(self, city='深圳', model_detail_slug='model_25023_cs', use_time=365, mile=2, ret_type='records'):
    #     """
    #     计算历史价格趋势
    #     """
    #     # 校验参数
    #     self.check_params_value(city, model_detail_slug, use_time, mile, category='history')
    #     # 计算时间
    #     times = [0, 30, 60, 90, 120, 150, 180]
    #     times_str = ['0', '-30', '-60', '-90', '-120', '-150', '-180']
    #     nums = 6
    #     if use_time < 181:
    #         times = []
    #         times_str = []
    #         nums = int((use_time - 1) / 30)
    #         for i in range(0, nums+1):
    #             times.append(i*30)
    #             times_str.append(str(i*30))
    #     # 计算车商交易价,车商收购价的历史价格走势
    #     data_buy = []
    #     data_sell = []
    #     data_private = []
    #     for ut in times:
    #         temp = self.predict(city, model_detail_slug, use_time-ut, mile)
    #         data_buy.append(temp.loc[(temp['intent'] == 'buy'), 'good'].values[0])
    #         data_sell.append(temp.loc[(temp['intent'] == 'sell'), 'good'].values[0])
    #         data_private.append(temp.loc[(temp['intent'] == 'private'), 'good'].values[0])
    #
    #     data_buy = process_unreasonable_history_price(data_buy, nums)
    #     data_sell = process_unreasonable_history_price(data_sell, nums)
    #     data_private = process_unreasonable_history_price(data_private, nums)
    #     result_b_2_c = pd.DataFrame([data_buy], columns=times_str)
    #     result_b_2_c['type'] = 'buy'
    #     result_c_2_b = pd.DataFrame([data_sell], columns=times_str)
    #     result_c_2_b['type'] = 'sell'
    #     result_c_2_c = pd.DataFrame([data_private], columns=times_str)
    #     result_c_2_c['type'] = 'private'
    #
    #     result = result_b_2_c.append(result_c_2_b, ignore_index=True)
    #     result = result.append(result_c_2_c, ignore_index=True)
    #
    #     if ret_type == 'records':
    #         return result.to_dict('records')
    #     else:
    #         return result
    #
    # def future_price_trend(self, city='深圳', model_detail_slug='model_25023_cs', use_time=365, mile=2, ret_type='records'):
    #     """
    #     计算未来价格趋势
    #     """
    #     # 校验参数
    #     self.check_params_value(city, model_detail_slug, use_time, mile, category='future')
    #     # 计算个人交易价的未来价格趋势
    #     data_buy = []
    #     data_sell = []
    #     data_private = []
    #     for ut in [0, 365, 720, 1095]:
    #         temp = self.predict(city, model_detail_slug, use_time+ut, mile)
    #         data_buy.append(temp.loc[(temp['intent'] == 'buy'), 'good'].values[0])
    #         data_sell.append(temp.loc[(temp['intent'] == 'sell'), 'good'].values[0])
    #         data_private.append(temp.loc[(temp['intent'] == 'private'), 'good'].values[0])
    #
    #     data_buy = process_unreasonable_future_price(data_buy)
    #     data_sell = process_unreasonable_future_price(data_sell)
    #     data_private = process_unreasonable_future_price(data_private)
    #     result_b_2_c = pd.DataFrame([data_buy], columns=['0', '365', '720', '1095'])
    #     result_b_2_c['type'] = 'buy'
    #     result_c_2_b = pd.DataFrame([data_sell], columns=['0', '365', '720', '1095'])
    #     result_c_2_b['type'] = 'sell'
    #     result_c_2_c = pd.DataFrame([data_private], columns=['0', '365', '720', '1095'])
    #     result_c_2_c['type'] = 'private'
    #
    #     result = result_b_2_c.append(result_c_2_b, ignore_index=True)
    #     result = result.append(result_c_2_c, ignore_index=True)
    #
    #     if ret_type == 'records':
    #         return result.to_dict('records')
    #     else:
    #         return result


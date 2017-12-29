import logging
from time import perf_counter

from valuate.predict import *

LOGGER = logging.getLogger()

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


def cal_intent_condition(prices, price_bn):
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


def process_mile(price, use_time, mile):
    """
    mile处理
    """
    # 正常行驶的车辆以一年2.5万公里为正常基数，低于2.5万公里的价格的浮动在+3.5%以内
    # 大于2.5万公里的若每年的平均行驶里程大于2.5万公里小于5万公里价格浮动在-3.5-7.5%
    # 若年平均形式里程大于5万公里及以上影响价格在-7.5-12.5%之间
    mile_per_month = mile / use_time
    if mile_per_month < gl.MILE_THRESHOLD_2_5:
        return price + 0.035 * (1 - mile_per_month/gl.MILE_THRESHOLD_2_5) * price
    elif gl.MILE_THRESHOLD_2_5 <= mile_per_month < gl.MILE_THRESHOLD_5:
        return price - (0.04 * (mile_per_month/gl.MILE_THRESHOLD_5)+0.035) * price
    elif gl.MILE_THRESHOLD_5 <= mile_per_month < gl.MILE_THRESHOLD_10:
        return price - (0.05 * (mile_per_month/gl.MILE_THRESHOLD_5)+0.075) * price
    else:
        return price - 0.125 * price


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


def process_unreasonable_future_price(data, nums):
    """
    处理不合理未来价格趋势
    """
    temp = data[1:]
    temp.sort(reverse=True)
    for i, value in enumerate(temp):
        data[i+1] = temp[i]

    for i in range(0, nums):
        if data[i] <= data[i + 1]:
            data[i + 1] = int(data[i] * 0.9)

    return data


def process_fill_zero(hedge):
    temp = hedge
    if len(hedge) < 18:
        for i in range(0, (18-len(hedge))):
            temp = '0'+ temp
    return temp


def predict_from_db(model_detail_slug, city, use_time):
    """
    从生产库查询预测
    """
    # 查找city和model_detail_slug编号
    city_id = province_city_map.loc[city, 'city_id']
    model_detail_slug_id = model_detail_map.loc[model_detail_slug, 'final_model_detail_slug_id']
    # 计算查询字段编号和月编号
    if (use_time % 6) == 0:
        column_num = str(int(use_time / 6) - 1)
        month_num = 6
    else:
        column_num = str(int(use_time / 6))
        month_num = use_time % 6
    # 查询
    record = db_operate.query_valuate(model_detail_slug_id, city_id, column_num, use_time)
    # 查找对应值
    dealer_hedge = str(record.loc[0, 'b2c_year_'+column_num])
    dealer_hedge = process_fill_zero(dealer_hedge)
    dealer_hedge = dealer_hedge[(month_num-1)*3:month_num*3]
    dealer_hedge = int(dealer_hedge) / 1000
    cpersonal_hedge = str(record.loc[0, 'c2c_year_'+column_num])
    cpersonal_hedge = process_fill_zero(cpersonal_hedge)
    cpersonal_hedge = cpersonal_hedge[(month_num-1)*3:month_num*3]
    cpersonal_hedge = int(cpersonal_hedge) / 1000
    return dealer_hedge, cpersonal_hedge


def process_prices_relate(dealer_price, cpersonal_price):
    """
    人工处理三类价格的相关性
    """
    buy = dealer_price
    private = cpersonal_price

    # 计算buy与private的比例关系
    private_buy_rate = (buy - private) / private
    # 人工处理预测不合理的三类价格
    if (private_buy_rate < 0) | (abs(private_buy_rate) > 0.12):
        private = int(buy * (1 - 0.0875))

    sell = int(private * (1 - 0.0525))
    return buy, private, sell


def process_adjust_profit(model_detail_slug, popularity):
    """
    调整值调整
    """
    index = str(model_detail_slug)+'_'+str(popularity)
    if index in model_detail_slug_popularity_index:
        rate = adjust_profit.loc[index, 'rate']
    else:
        rate = 0
    return rate


def check_params_value(city, model_detail_slug, use_time, mile, category):
    """
    校验参数
    """
    # 校验city
    if city not in cities:
        raise ApiParamsValueError('city', city, 'Unknown city!')
    # 校验model
    if model_detail_slug not in models:
        raise ApiParamsValueError('model_detail_slug', model_detail_slug, 'Unknown model!')
    # 校验mile
    if not ((isinstance(mile, int)) | (isinstance(mile, float))):
        raise ApiParamsTypeError('mile', mile, 'Mile must be int or float!')
    elif mile < 0:
        raise ApiParamsValueError('mile', mile, 'Mile must be greater than zero!')
    # 校验use_time
    if not isinstance(use_time, int):
        raise ApiParamsTypeError('use_time', use_time, 'Use_time must be int!')
    if category == 'valuate':
        if (use_time < 1) | (use_time > 240):
            raise ApiParamsValueError('use_time', use_time, 'The use_time of Forecast must be in 1-240!')
    elif category == 'history':
        if (use_time < 1) | (use_time > 240):
            raise ApiParamsValueError('use_time', use_time, 'The use_time of historical trend must be in 1-240!')
    elif category == 'future':
        if (use_time < 1) | (use_time > 240):
            raise ApiParamsValueError('use_time', use_time, 'The use_time of future trend must be in 1-240!')


class Predict(object):

    def __init__(self):
        """
        加载各类匹配表和模型
        """
        self.result = []
        self.valuate_model = []

    def add_process_intent(self, buy, private, sell, popularity, price_bn):
        """
        根据交易方式修正预测值
        """
        # 组合结果
        self.result = result_map.copy()
        self.result.loc[(self.result['intent'] == 'buy'), 'predict_price'] = buy
        self.result.loc[(self.result['intent'] == 'private'), 'predict_price'] = private
        self.result.loc[(self.result['intent'] == 'sell'), 'predict_price'] = sell
        self.result['predict_price'] = self.result['predict_price'].fillna(buy)

        self.result['popularity'] = popularity
        self.result['profit_rate'] = self.result.apply(process_profit_rate, axis=1)
        self.result['buy_profit_rate'] = self.result.apply(process_buy_profit_rate, axis=1)
        self.result['predict_price'] = self.result['predict_price'] / self.result['buy_profit_rate']
        self.result['predict_price'] = self.result['profit_rate'] * self.result['predict_price']

        # 计算所有交易类型
        self.result = cal_intent_condition(self.result.predict_price.values, price_bn)

    def predict(self, city='深圳', model_detail_slug='model_25023_cs', use_time=12, mile=2, ret_type='records'):
        """
        预测返回
        """
        # 校验参数
        check_params_value(city, model_detail_slug, use_time, mile, category='valuate')

        # 查找款型对应的新车指导价,调整后的款型
        price_bn = model_detail_map.loc[model_detail_slug, 'final_price_bn']
        price_bn = price_bn * 10000
        province = province_city_map.loc[city, 'province']
        model_slug = model_detail_map.loc[model_detail_slug, 'model_slug']
        final_model_detail_slug = model_detail_map.loc[model_detail_slug, 'final_model_detail_slug']

        # 预测返回保值率
        t = perf_counter()
        dealer_hedge, cpersonal_hedge = predict_from_db(final_model_detail_slug, city, use_time)
        elapsed_ms = round(((perf_counter() - t) * 1000), 2)
        LOGGER.info('Read-db-elaspsed: %.2f' % elapsed_ms)
        dealer_price, cpersonal_price = dealer_hedge * price_bn, cpersonal_hedge * price_bn

        # 处理mile
        dealer_price = process_mile(dealer_price, use_time, mile)
        cpersonal_price = process_mile(cpersonal_price, use_time, mile)

        # 处理价格之间的相关性
        buy, private, sell = process_prices_relate(dealer_price, cpersonal_price)

        # 获取流行度
        index = str(model_slug)+'_'+str(province)
        if index in province_popularity_index:
            popularity = province_popularity_map.loc[index, 'popularity']
        else:
            popularity = 'C'

        # 进行调整值最终调整
        rate = process_adjust_profit(model_detail_slug, popularity)
        buy, private, sell = buy*(1+rate), private*(1+rate), sell*(1+rate)

        # 根据交易方式修正预测值
        self.add_process_intent(buy, private, sell, popularity, price_bn)

        if ret_type == 'records':
            return self.result.to_dict('records')
        else:
            return self.result

    def history_price_trend(self, city='深圳', model_detail_slug='model_25023_cs', use_time=12, mile=2, ret_type='records'):
        """
        计算历史价格趋势
        """
        # 校验参数
        check_params_value(city, model_detail_slug, use_time, mile, category='history')
        # 计算时间
        times = [0, 1, 2, 3, 4, 5, 6]
        times_str = ['0', '-1', '-2', '-3', '-4', '-5', '-6']
        nums = 6
        if use_time <= 6:
            times = []
            times_str = []
            nums = use_time-1
            for i in range(0, nums+1):
                times.append(i)
                times_str.append(str(-i))
        # 计算车商交易价,车商收购价的历史价格走势
        data_buy = []
        data_sell = []
        data_private = []
        for ut in times:
            temp = self.predict(city, model_detail_slug, use_time-ut, mile, ret_type='normal')
            data_buy.append(temp.loc[(temp['intent'] == 'buy'), 'good'].values[0])
            data_sell.append(temp.loc[(temp['intent'] == 'sell'), 'good'].values[0])
            data_private.append(temp.loc[(temp['intent'] == 'private'), 'good'].values[0])

        data_buy = process_unreasonable_history_price(data_buy, nums)
        data_sell = process_unreasonable_history_price(data_sell, nums)
        data_private = process_unreasonable_history_price(data_private, nums)
        result_b_2_c = pd.DataFrame([data_buy], columns=times_str)
        result_b_2_c['type'] = 'buy'
        result_c_2_b = pd.DataFrame([data_sell], columns=times_str)
        result_c_2_b['type'] = 'sell'
        result_c_2_c = pd.DataFrame([data_private], columns=times_str)
        result_c_2_c['type'] = 'private'

        result = result_b_2_c.append(result_c_2_b, ignore_index=True)
        result = result.append(result_c_2_c, ignore_index=True)

        if ret_type == 'records':
            return result.to_dict('records')
        else:
            return result

    def future_price_trend(self, city='深圳', model_detail_slug='model_25023_cs', use_time=365, mile=2, ret_type='records'):
        """
        计算未来价格趋势
        """
        # 校验参数
        check_params_value(city, model_detail_slug, use_time, mile, category='future')
        # 计算时间
        times = [0, 12, 24, 36]
        times_str = ['0', '12', '24', '36']
        nums = 3
        if use_time > 204:
            times = []
            times_str = []
            nums = int((240-use_time) / 12)
            for i in range(0, nums+1):
                times.append(i*12)
                times_str.append(str(i*12))
        # 计算个人交易价的未来价格趋势
        data_buy = []
        data_sell = []
        data_private = []
        for ut in times:
            temp = self.predict(city, model_detail_slug, use_time+ut, mile, ret_type='normal')
            data_buy.append(temp.loc[(temp['intent'] == 'buy'), 'good'].values[0])
            data_sell.append(temp.loc[(temp['intent'] == 'sell'), 'good'].values[0])
            data_private.append(temp.loc[(temp['intent'] == 'private'), 'good'].values[0])

        data_buy = process_unreasonable_future_price(data_buy, nums)
        data_sell = process_unreasonable_future_price(data_sell, nums)
        data_private = process_unreasonable_future_price(data_private, nums)
        result_b_2_c = pd.DataFrame([data_buy], columns=times_str)
        result_b_2_c['type'] = 'buy'
        result_c_2_b = pd.DataFrame([data_sell], columns=times_str)
        result_c_2_b['type'] = 'sell'
        result_c_2_c = pd.DataFrame([data_private], columns=times_str)
        result_c_2_c['type'] = 'private'

        result = result_b_2_c.append(result_c_2_b, ignore_index=True)
        result = result.append(result_c_2_c, ignore_index=True)

        if ret_type == 'records':
            return result.to_dict('records')
        else:
            return result


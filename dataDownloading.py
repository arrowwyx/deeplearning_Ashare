import akshare as ak
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def get_stock_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取start到end之间的所有A股的日期，开盘，最高，最低，收盘，成交量，vwap
    :param start_date: 如格式"20240226"
    :param end_date: 如格式"20240226"
    :return: pd.Dataframe
    """
    stock_list = ak.stock_zh_a_spot_em()['代码'].tolist()
    all_data = []

    for code in tqdm(stock_list, desc="Downloading"):
        try:
            stock_data = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date,
                                            adjust="hfq")
            stock_data = stock_data[['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额']]
            stock_data['code'] = code
            all_data.append(stock_data)
        except Exception as e:
            continue

    all_stocks = pd.concat(all_data)
    all_stocks.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close',
                               '成交量': 'volume', '成交额': 'amount'}, inplace=True)
    all_stocks['date'] = pd.to_datetime(all_stocks['date']).dt.strftime('%Y%m%d')
    all_stocks['vwap'] = all_stocks['amount'] / all_stocks['volume']
    all_stocks.drop(columns=['amount'], inplace=True)
    all_stocks.set_index(['date', 'code'], inplace=True)
    all_stocks.sort_index(inplace=True)
    return all_stocks


def calc_label(df: pd.DataFrame, window) -> pd.DataFrame:
    """
    用open收益率计算label
    默认不会把前几天的nan值drop
    :param df: 格式如get_stock_data返回值的Dataframe
    :param window: 计算几天的收益率
    :return:
    """
    label = df[['open']].groupby(level='code', group_keys=False).apply(lambda x: x.pct_change(periods=window)
                                                                       .shift(-(window + 2))).rename(
        columns={'open': f'ret_{window}'})
    # 对label做截面标准化
    label = label.groupby(level='date', group_keys=False).apply(lambda x: x.apply(zscore_group))
    return label


def make_window(df: pd.DataFrame, window_len: int) -> pd.DataFrame:
    df_unstack = df.unstack(1)
    df_list = []
    for i in range(window_len):
        df_list.append(df_unstack.shift(i).stack())
    df_concat = pd.concat(df_list, axis=1)
    df_concat.columns = pd.MultiIndex.from_product([list(range(window_len)), list(df.columns)])
    df_concat = df_concat.sort_index().swaplevel(axis=1)
    # 遍历每个特征，用最后一天的值做时序标准化
    for feature in df.columns:
        last_day_value = df[feature]
        for day in range(window_len):
            df_concat[(feature, day)] = df_concat[(feature, day)]/last_day_value
    # 再对于每天，做截面z-score标准化
    df_zscore = df_concat.groupby(level='date', group_keys=False).apply(lambda x: x.apply(zscore_group))
    return df_zscore


def zscore_group(df_group):
    # 计算标准差
    std = df_group.std()
    # 如果标准差为零，直接返回原数据
    if std == 0:
        return df_group
    # 否则，执行 Z-分数标准化
    else:
        return (df_group - df_group.mean()) / std


if __name__ == "__main__":
    data = pd.read_hdf("Data/2020_2023kBar.h5")
    label = calc_label(data, 10)
    # data.to_hdf("Data/2020_2023kBar.h5", key='data')
    label.to_hdf("Data/2020_2023ret_10.h5", key='label')
    # df = pd.read_hdf('Data/2020_2023kBar.h5')
    # window = make_window(df, 5)
    # window.to_hdf("Data/2020_2023window.h5", key='data')

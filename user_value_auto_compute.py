#!/usr/bin/env python
# coding=UTF-8

## ======================================================
## @ScriptName:      user_value_auto_compute.py
## @Author:          xiangyu.xu
## @DateTime:        2017-06-13 15:52:10
## @Description:     用户价值自动化计算
## @input:           输入参数，1：渠道名称，默认为'auto',自动计算所有的渠道
##                             2: 电商类型, 默认为'all',自动计算e_supplier_profit、ad_profit、licai_order_profit三种类型
##                             3: 开始时间，默认为14天前，可以直接指定
##                             4: 结束时间，默认为7天前，可以直接指定
## ======================================================


# import matplotlib.pyplot as plt
## %matplotlib inline
import numpy as np
import pandas as pd
import calendar
from pandas import DataFrame
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# from sqlalchemy import create_engine
#import pymssql
import subprocess
import time
import datetime
import sys
import os
import getopt

c_path = os.path.split(os.path.realpath(__file__))[0]

# 禁用一些警告
pd.options.mode.chained_assignment = None


##获取前一天的时间，格式%Y%m%d%H
def day_get(datestr, index):
    dt = datetime.datetime.strptime(datestr, "%Y-%m-%d")
    dt_next = dt + datetime.timedelta(days=index)
    return dt_next.strftime("%Y-%m-%d")


def generate_csv_source(data_file):
    #判断文件是否存在
    if not os.path.exists(data_file):
        # 拉取hive表数据
        cmd_sql = 'hive -e "set hive.exec.reducers.bytes.per.reducer=100000000;set hive.exec.reducers.max=200;set hive.cli.print.header=true; \
                select * from dm.user_value_auto_compute_base_info where ds<>\'\'  \
                and type in (\'e_supplier_profit\',\'licai_order_profit\',\'ad_profit\') \
                " >%s' % (data_file)
        subprocess.call(cmd_sql, shell=True)
        # 替换其中的字段分隔符/t为,
        cmd_sed = 'sed -i "s/\t/,/g" %s' % (data_file)
        subprocess.call(cmd_sed, shell=True)
        print "文件已生成："+data_file
    else:
        print "最新文件已存在："+data_file


def insert_to_table(data_cur, c_path, ds,freq_type):
    # data_cur.to_csv('./user_value_auto_compute_result.csv', index=False, header=None, encoding="utf8")
    path_result = '%s/user_value_auto_compute_result.csv' % (c_path)
    print '生成的结果csv文件:',path_result
    data_cur.to_csv(path_result, index=False,header=None, float_format = '%.4f', encoding="utf8")
    cmd_hive = '''hive -e "load data local inpath \'''' + c_path + '''/user_value_auto_compute_result.csv\' OVERWRITE  into table rpt.user_value_auto_compute_result partition(freq_type='%s',ds='%s')"''' % (freq_type,ds)
    # 导入hive库
    print cmd_hive
    subprocess.call(cmd_hive, shell=True)
    print '生成文件:user_value_auto_compute_result.csv'

# def insert_in_sqlserver_table(df):
#     # 建立 engine 並連線至 SQL Server
#     db_usr = 'un_xiangyu.xu'
#     db_pwd = 'xxy!@345678'
#     db_server = 'dwprd.db.51fanli.it'
#     db_name = 'dw'
#     engine = create_engine("mssql+pymssql://%s:%s@%s/%s?charset=utf8"
#                            % (db_usr, db_pwd, db_server, db_name), echo=True)
#     conn = engine.connect()
#     pd.io.sql.to_sql(df, 'user_value_auto_compute_result', conn, schema='rpt', if_exists='append')
#     conn.close()


# 字符串日期转成对应的年周
def change_date_week(datestr):
    d = datetime.datetime.strptime(datestr, '%Y-%m-%d')
    return d.strftime('%Y') + d.strftime('%W')


# 获取一周第一天
def get_firstday_of_week(datestr):
    d1 = datetime.datetime.strptime(datestr, '%Y-%m-%d')
    d2 = d1 - datetime.timedelta(days=d1.weekday())
    return d2.strftime('%Y-%m-%d')


# 获取一周最后一天
def get_lastday_of_week(datestr):
    d1 = datetime.datetime.strptime(datestr, '%Y-%m-%d')
    d2 = d1 + datetime.timedelta(days=6) - datetime.timedelta(days=d1.weekday())
    return d2.strftime('%Y-%m-%d')


# 字符串日期转成对应的周范围
def change_date_weekscope(datestr):
    return get_firstday_of_week(datestr) + '-' + get_lastday_of_week(datestr)


# 获取上上周的日期
def get_last_lastweek_day(datestr):
    d1 = datetime.datetime.strptime(datestr, '%Y-%m-%d')
    d2 = d1 - datetime.timedelta(days=14)
    return d2.strftime('%Y-%m-%d')

# 获取月初和月末
def get_month_begin_and_end(datestr):
    d1 = datetime.datetime.strptime(datestr, '%Y-%m-%d')
    begin = '%d-%02d-01' % (d1.year, d1.month)
    wday, monthRange = calendar.monthrange(d1.year, d1.month)
    end = '%d-%02d-%02d' % (d1.year, d1.month, monthRange)
    return (begin,end)

# 获取上月月末
def get_last_month_end(datestr):
    d1 = datetime.datetime.strptime(datestr, '%Y-%m-%d')
    first = datetime.date(day=1, month=d1.month, year=d1.year)
    lastMonth = first - datetime.timedelta(days=1)
    return lastMonth.strftime('%Y-%m-%d')

def get_date_list(bdate, edate):
    ds_list = {
        "ds": bdate.replace('-', '') + '-' + edate.replace('-', ''),
        "ds_0": bdate,
        "ds_1": edate,
        "ds_1m": day_get(bdate, -30 * 2 + 14).replace('-', '') + '-' + day_get(bdate, -30 * 1 + 14).replace('-', ''),
        "ds_1m_0": day_get(bdate, -30 * 2 + 14),
        "ds_1m_1": day_get(bdate, -30 * 1 + 14),
        "ds_3m": day_get(bdate, -30 * 6 + 14).replace('-', '') + '-' + day_get(bdate, -30 * 3 + 14).replace('-', ''),
        "ds_3m_0": day_get(bdate, -30 * 6 + 14),
        "ds_3m_1": day_get(bdate, -30 * 3 + 14),
        "ds_1y": day_get(bdate, -30 * 13 + 14).replace('-', '') + '-' + day_get(bdate, -30 * 12 + 14).replace('-', ''),
        "ds_1y_0": day_get(bdate, -30 * 13 + 14),
        "ds_1y_1": day_get(bdate, -30 * 12 + 14),
        
        "ds_1y3": get_month_begin_and_end(day_get(bdate, -30 * 15 + 14))[0].replace('-', '') + '-' + get_month_begin_and_end(day_get(bdate, -30 * 3 + 14))[0].replace('-', ''),
        "ds_1y3_0": get_month_begin_and_end(day_get(bdate, -30 * 15 + 14))[0],
        "ds_1y3_1": get_month_begin_and_end(day_get(bdate, -30 * 3 + 14))[0]
    }
    return ds_list


def by_linearRegression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
        a=0
        b=0
    else :
        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        LinearRegression(copy_X=True, fit_intercept=False, n_jobs=1, normalize=False)
        # y=ax+b
        # 获取系数和截距
        a = linreg.coef_[0][0]
        b = linreg.intercept_[0]
    return a, b


def add_columns(data):
    # 加字段：七天购物率，人均月毛利，人均年毛利
    data['rate_d07'] = data['isprofit_d07_cnt'] / data['reg_user_cnt']
    data['profit_m01_avg'] = data['profit_m01'] / data['reg_user_cnt']
    data['profit_y01_avg'] = data['profit_y01'] / data['reg_user_cnt']
    return data



# class MSSQL:
#     def __init__(self, host, user, pwd, db):
#         self.host = host
#         self.user = user
#         self.pwd = pwd
#         self.db = db

#     def GetConnect(self):
#         self.conn = pymssql.connect(host=self.host, user=self.user, password=self.pwd, database=self.db, charset="utf8")
#         cur = self.conn.cursor()

#         return cur

#     def ExecQuery(self, sql):
#         cur = self.GetConnect()
#         cur.execute(sql)
#         resList = cur.fetchall()

#         cur.close()
#         self.conn.close()
#         return resList

#     def Execute(self, sql):
#         cur = self.GetConnect()
#         try:
#             cur.execute(sql)
#         except Exception, e:
#             self.conn.rollback()
#             print (e)
#         self.conn.commit()
#         self.conn.close()

#     def ExecuteBatch(self, table_name, col, wildcards, data):
#         cur = self.GetConnect()
#         s = "INSERT INTO %s (%s) VALUES(%s)"
#         print s
#         s = s % (table_name, col, wildcards)
#         print s
#         cur.executemany(s, data)
#         self.conn.commit()
#         self.conn.close()


# def get_sqlserver_table(df, ds):
#     ms = MSSQL(host="dwprd.db.51fanli.it", user="un_xiangyu.xu", pwd="xxy!@345678", db="dw")
#     table_name = 'rpt.user_value_auto_compute_result'
#     delete_sql = "delete from %s where ds = '%s'" % (table_name, ds)
#     print delete_sql
#     ms.Execute(delete_sql)
#     wildcards = ','.join(['%s'] * len(df.columns))
#     data = [tuple(x) for x in df.values]
#     col = 'ds, column_name, channel, reg_user_cnt_supplier, user_value_all, type_name_supplier, rate_d07_supplier, user_value_supplier, m_w_ds_supplier, m_w_coef_supplier, m_w_intercept_supplier, y_m_ds_supplier, y_m_multiple_supplier, type_name_licai, profit_d07_licai, user_value_licai, m_w_ds_licai, m_w_multiple_licai, y_m_ds_licai, y_m_multiple_licai, type_name_ad, profit_d07_ad, user_value_ad, m_w_ds_ad, m_w_multiple_ad, y_m_ds_ad, y_m_multiple_ad'
#     ms.ExecuteBatch(table_name, col, wildcards, data)

def delete_exception_value(df):
    ##根据人均首月毛利去除异常值，3倍标准差之外的去掉，注册数小于50的去掉
    data = df[(np.abs((df['profit_m01'] / df['reg_user_cnt']) - (
        df['profit_m01'] / df['reg_user_cnt']).std()) <= 3 * np.mean(
        df['profit_m01'] / df['reg_user_cnt'])) & (df['reg_user_cnt'] > 0)]
    # data.to_csv('./data_1234.csv', index=False, encoding="gb18030")
    return data

# 计算系数，包括月周比（斜率a、截距b）、年月比
# 计算电商的系数(已到最小的需要计算的类别了)
def get_coef_of_e_supplier(data, channel_class, channel_category, channel_department, ds_list):
    data = data.groupby(
        ['channel_class_value_refer', 'channel_category_value_refer', 'channel_department_value_refer', 'all_refer',
         'type', 'ds']).sum()
    data = data.reset_index()
    data = add_columns(data)

    # 获取首月和七天购物率的关系
    # 首月(T-90至T-180的注册用户)
    data_1_month_ago = data[(data['ds'] >= ds_list['ds_3m_0']) & (data['ds'] <= ds_list['ds_3m_1'])]
    if data_1_month_ago.empty:
        a=0
        b=0
        print "Empty data:",channel_class, channel_category, channel_department
    else :
        a, b = by_linearRegression(data_1_month_ago[['rate_d07']], data_1_month_ago[['profit_m01_avg']])

    # 全年(T-360到T-390的注册用户)，取全年的毛利（前8个月毛利+（6、7、8三个月毛利均值*4））
    data_1year_ago = data[(data['ds'] >= ds_list['ds_1y_0']) & (data['ds'] <= ds_list['ds_1y_1'])]
    # 全年毛利
    profit_y01 = data_1year_ago['profit_pm08'].sum() + data_1year_ago['profit_mm03'].sum() * 4 / 3
    profit_m01 = data_1year_ago['profit_m01'].sum()
    # 倍数关系
    if profit_m01 > 0:
        x = profit_y01 * 1.0 / profit_m01
    else:
        x = 0
    # 单独计算参考系数的用户价值
    data_1_week_ago = data[(data['ds'] >= ds_list['ds_0']) & (data['ds'] <= ds_list['ds_1'])]
    reg_user_cnt = data_1_week_ago['reg_user_cnt'].sum()
    isprofit_d07_cnt = data_1_week_ago['isprofit_d07_cnt'].sum()
    if reg_user_cnt > 0:
        rate_d07 = isprofit_d07_cnt * 1.0 / reg_user_cnt
    else:
        rate_d07 = 0
    # 用户价值：v=(a*rate_d07+b)*x
    if reg_user_cnt == 0:
        v = 0
    else:
        v = (a * rate_d07 + b) * x * 1.0
    # 分别输出：计算时间段、注册数、七天购物率、用户价值、月周比参考时间段、月周比系数、月周比截距、年月比参考时间段、年月比倍数、
    # print '月周比参考时间段:'+a_period_1
    # print '    月周比方程:'+'y='+str(a)+'x+'+str(b)
    # print '年月比参考时间段:'+a_period_2
    # print '    年月比方程:'+'y='+str(x)+'x'
    # return [a_period_0,reg_user_cnt,rate_d07,v,a_period_1,a,b,a_period_2,x]
    ##'ds','reg_user_cnt','rate_d07','user_value','m_w_ds','m_w_coef','m_w_intercept','y_m_ds','y_m_multiple'
    return {"ds": ds_list['ds'],
            "reg_user_cnt": reg_user_cnt,
            "rate_d07": rate_d07,
            "user_value": v,
            "m_w_ds": ds_list['ds_3m'],
            "m_w_coef": a,
            "m_w_intercept": b,
            "y_m_ds": ds_list['ds_1y'],
            "y_m_multiple": x
            }


# 计算广告、理财的系数
def get_coef_of_ad_or_licai(data, channel_class, channel_category, channel_department, ds_list):
    data = data.groupby(
        ['channel_class_value_refer', 'channel_category_value_refer', 'channel_department_value_refer', 'all_refer',
         'type', 'ds']).sum()
    data = data.reset_index()
    data = add_columns(data)

    # 获取首月和七天毛利的倍数关系
    # 首月(T-30至T-60的注册用户)
    data_1_month_ago = data[(data['ds'] >= ds_list['ds_1m_0']) & (data['ds'] <= ds_list['ds_1m_1'])]
    profit_m01_1m = data_1_month_ago['profit_m01'].sum()
    profit_d07_1m = data_1_month_ago['profit_d07'].sum()
    # 倍数关系
    if profit_d07_1m > 0:
        x1 = profit_m01_1m / profit_d07_1m
    else:
        x1 = 0
    # 全年(T-360到T-390的注册用户)，取全年的毛利（前8个月毛利+（6、7、8三个月毛利均值*4））

    if ds_list['ds_1y_0'] < '2016-07-14':
        ds_list['ds_1y_0'] = '2016-07-14'
        ds_list['ds_1y_1'] = '2016-08-13'
        ds_list['ds_1y'] = '20160714-20160813'
    data_1year_ago = data[(data['ds'] >= ds_list['ds_1y_0']) & (data['ds'] <= ds_list['ds_1y_1'])]
    # 全年毛利
    profit_y01 = data_1year_ago['profit_y01'].sum()
    profit_m01 = data_1year_ago['profit_m01'].sum()
    # 倍数关系
    if profit_m01 > 0:
        x2 = profit_y01 * 1.0 / profit_m01
    else:
        x2 = 0
    # 单独计算参考系数的用户价值
    data_1_week_ago = data[(data['ds'] >= ds_list['ds_0']) & (data['ds'] <= ds_list['ds_1'])]
    reg_user_cnt = data_1_week_ago['reg_user_cnt'].sum()
    profit_d07 = data_1_week_ago['profit_d07'].sum()
    # 用户价值：v = (profit_d07*x1)*x2*1.0/reg_user_cnt
    if reg_user_cnt > 0:
        v = (profit_d07 * x1) * x2 * 1.0 / reg_user_cnt
    else:
        v = 0
    # 分别输出：计算时间段、注册数、七天购物率、用户价值、月周比参考时间段、月周比倍数、年月比参考时间段、年月比倍数、
    # print '月周比参考时间段:'+a_period_1
    # print '    月周比方程:'+'y='+str(x1)+'x'
    # print '年月比参考时间段:'+a_period_2
    # print '    年月比方程:'+'y='+str(x2)+'x'
    # return [a_period_0,reg_user_cnt,profit_d07,v,a_period_1,x1,a_period_2,x2]
    return {"ds": ds_list['ds'],
            "reg_user_cnt": reg_user_cnt,
            "profit_d07": profit_d07,
            "user_value": v,
            "m_w_ds": ds_list['ds_1m'],
            "m_w_multiple": x1,
            "y_m_ds": ds_list['ds_1y'],
            "y_m_multiple": x2
            }


# 循环获取维度列，生成一个一个带计算系数的dataframe：
# 包含字段：['type','column_name','channel','ds','reg_user_cnt','rate_d07','user_value','m_w_ds','m_w_coef','m_w_intercept','y_m_ds','y_m_multiple']
# 定义：     计算时间段、注册数、七天购物率、用户价值、月周比参考时间段、月周比系数、月周比截距、年月比参考时间段、年月比倍数
def get_different_coef_by_class(data_all, type_name, ds_list):
    # 定义参考的维度列，后期用户价值需要根据此计算
    classfications = ['channel_class_value_refer', 'channel_category_value_refer', 'channel_department_value_refer',
                      'all_refer']
    xishu_table = []
    for index, row in data_all[(data_all['ds'] >= ds_list['ds_0']) & (data_all['ds'] <= ds_list['ds_1']) & (
                data_all['type'] == type_name)][classfications].drop_duplicates().iterrows():
        channel_class, channel_category, channel_department, all_compute = row['channel_class_value_refer'], row[
            'channel_category_value_refer'], row['channel_department_value_refer'], row['all_refer']
        data = data_all[(data_all['channel_class_value_refer'] == data_all['channel_class']) &
            (data_all['channel_class'] == channel_class) & (
            data_all['channel_category'] == channel_category) & (
                            data_all['channel_department'] == channel_department) & (
                            data_all['all_compute'] == all_compute) & (data_all['type'] == type_name)]
        data = delete_exception_value(data)
        if data.empty:
            continue
        if type_name in ['e_supplier_profit', 'common_order_profit', 'tao2_order_profit', 'zy_order_profit']:
            coef_dict = get_coef_of_e_supplier(data, channel_class, channel_category, channel_department, ds_list)
        if type_name in ['ad_profit', 'ad_common_profit', 'ad_licai_profit', 'licai_order_profit']:
            coef_dict = get_coef_of_ad_or_licai(data, channel_class, channel_category, channel_department, ds_list)
        head_col = {
            "type_name": type_name,
            "column_name": '',
            "channel": channel_class,
            "channel_category": channel_category,
            "channel_department": channel_department
        }
        coef_dict = dict(head_col.items() + coef_dict.items())
        xishu_table.append(coef_dict)
    df = DataFrame(xishu_table)
    df = df.drop_duplicates()
    return df


# 根据给定的渠道查询参考的系数
def get_xishu_from_table(df, refers):
    # 取出系数表里的用户价值，用于判断是否需要获取上一级类别系数
    v = df[(df['channel'] == refers[1]) & (df['channel_category'] == refers[2]) & (
        df['channel_department'] == refers[3]) & (df['type_name'] == refers[0])]
    if v['user_value'].empty:
        v = df[(df['channel'] == refers[2]) & (df['channel_category'] == refers[2]) & (
            df['channel_department'] == refers[3]) & (df['type_name'] == refers[0])]
        if v['user_value'].empty:
            v = df[(df['channel'] == refers[3]) & (df['channel_category'] == refers[3]) & (
                df['channel_department'] == refers[3]) & (df['type_name'] == refers[0])]
            if v['user_value'].empty:
                v = df[(df['channel'] == refers[4]) & (df['channel_category'] == refers[4]) & (
                    df['channel_department'] == refers[4]) & (df['type_name'] == refers[0])]
    # return v['m_w_coef'].values[0],v['m_w_intercept'].values[0],v['y_m_multiple'].values[0],v['m_w_ds'].values[0],v['y_m_ds'].values[0]
    return v.to_dict(orient='records')[0]

def optimized_coef_table(df):
    ##优化系数表
    ##针对系数表中年月比或者月周比（系数）小于等于1的记录，修改其年月比为其父类的年月比
    xishu_table = []
    for index, row in df.iterrows():
        if row['y_m_multiple']<=1:
            for i, r in df.iterrows():
                if (row['channel_category'] == r['channel']) and (row['channel_category'] == r['channel_category']) and (row['channel_department'] == r['channel_department']) and (row['ds'] == r['ds']):
                    row['y_m_multiple'] = r['y_m_multiple']

        if row['type_name'] in ['e_supplier_profit', 'common_order_profit', 'tao2_order_profit','zy_order_profit']:
            if row['m_w_coef']<=1:
                for i, r in df.iterrows():
                    if (row['channel_category'] == r['channel']) and (row['channel_category'] == r['channel_category']) and (row['channel_department'] == r['channel_department']) and (row['ds'] == r['ds']):
                        row['m_w_coef'] = r['m_w_coef']
                        row['m_w_intercept'] = r['m_w_intercept']
        elif row['type_name'] in ['ad_profit', 'ad_common_profit', 'ad_licai_profit', 'licai_order_profit']:
            if row['m_w_multiple']<=1:
                for i, r in df.iterrows():
                    if (row['channel_category'] == r['channel']) and (row['channel_category'] == r['channel_category']) and (row['channel_department'] == r['channel_department']) and (row['ds'] == r['ds']):
                        row['m_w_multiple'] = r['m_w_multiple']
        xishu_table.append(row)
    res = DataFrame(xishu_table)
    res = res.drop_duplicates()
    return res

# ,这里只计算一个类别
def compute_user_value(coef_table, data_1_week_ago, type_name, ds_list, refers):
    reg_user_cnt = data_1_week_ago['reg_user_cnt'].sum()
    isprofit_d07_cnt = data_1_week_ago['isprofit_d07_cnt'].sum()
    profit_d07 = data_1_week_ago['profit_d07'].sum()
    if reg_user_cnt > 0:
        rate_d07 = isprofit_d07_cnt * 1.0 / reg_user_cnt
    else:
        rate_d07 = 0
    # 参考的渠道
    # print refers
    # 通过参考表获取系数
    # a,b,x,m_w_ds,y_m_ds=get_xishu_from_table(coef_table,refers)
    ref_dict = get_xishu_from_table(coef_table, refers)
    # 用户价值：v=(a*rate_d07+b)*x
    if reg_user_cnt == 0:
        v = 0
    else:
        if type_name in ['e_supplier_profit', 'common_order_profit', 'tao2_order_profit', 'zy_order_profit']:
            # v = (a*rate_d07+b)*x*1.0
            v = (ref_dict['m_w_coef'] * rate_d07 + ref_dict['m_w_intercept']) * ref_dict['y_m_multiple'] * 1.0
            return_dict = {"ds": ds_list['ds'],
                           "reg_user_cnt": reg_user_cnt,
                           "rate_d07": rate_d07,
                           "user_value": v,
                           "m_w_ds": ref_dict['m_w_ds'],
                           "m_w_coef": ref_dict['m_w_coef'],
                           "m_w_intercept": ref_dict['m_w_intercept'],
                           "y_m_ds": ref_dict['y_m_ds'],
                           "y_m_multiple": ref_dict['y_m_multiple']
                           }
        if type_name in ['ad_profit', 'ad_common_profit', 'ad_licai_profit', 'licai_order_profit']:
            v = (profit_d07 * ref_dict['m_w_multiple']) * ref_dict['y_m_multiple'] * 1.0 / reg_user_cnt
            return_dict = {"ds": ds_list['ds'],
                           "reg_user_cnt": reg_user_cnt,
                           "profit_d07": profit_d07,
                           "user_value": v,
                           "m_w_ds": ref_dict['m_w_ds'],
                           "m_w_multiple": ref_dict['m_w_multiple'],
                           "y_m_ds": ref_dict['y_m_ds'],
                           "y_m_multiple": ref_dict['y_m_multiple']
                           }
    # return type_name,a_period_0,reg_user_cnt,rate_d07,v,m_w_ds,a,b,y_m_ds,x
    # 重构ref_dict，作为返回
    return return_dict


# 循环获取带计算的小类别渠道，分别计算用户价值
def get_user_values(data_all, coef_table, type_name, ds_list):
    # 定义参考的维度列，后期用户价值需要根据此计算
    classfications = ['channel_class', 'channel_category', 'channel_department', 'all_compute']
    xishu_table = []

    for index, row in data_all[(data_all['ds'] >= ds_list['ds_0']) & (data_all['ds'] <= ds_list['ds_1'])][
        classfications].drop_duplicates().iterrows():
        channel_class, channel_category, channel_department, all_compute = row['channel_class'], row[
            'channel_category'], row['channel_department'], row['all_compute']

        data = data_all[
            (data_all['channel_class'] == channel_class) & (data_all['channel_category'] == channel_category) & (
                data_all['channel_department'] == channel_department) & (data_all['all_compute'] == all_compute) & (
                data_all['type'] == type_name)]

        channel_class_value_refer, channel_category_value_refer, channel_department_value_refer, all_refer = data[
                             'channel_class_value_refer'].drop_duplicates().values[
                             0], \
                         data[
                             'channel_category_value_refer'].drop_duplicates().values[
                             0], \
                         data[
                             'channel_department_value_refer'].drop_duplicates().values[
                             0], \
                         data[
                             'all_refer'].drop_duplicates().values[
                             0]
        refers = [type_name, channel_class_value_refer, channel_category_value_refer, channel_department_value_refer,
                  all_refer]

        # type_name,a_period_0,reg_user_cnt,rate_d07,v,m_w_ds,a,b,y_m_ds,x = compute_user_value(coef_table,data,type_name)
        special_dict = compute_user_value(coef_table, data, type_name, ds_list, refers)
        # datas.append(data)
        head_col = {
            "type_name": type_name,
            "column_name": '',
            "channel": channel_class,
            "channel_category": channel_category,
            "channel_department": channel_department
        }
        special_dict = dict(head_col.items() + special_dict.items())
        xishu_table.append(special_dict)
    df = DataFrame(xishu_table)
    # df.columns = ['type','column_name','channel','ds','reg_user_cnt','rate_d07','user_value','m_w_ds','m_w_coef','m_w_intercept','y_m_ds','y_m_multiple']
    df = df.drop_duplicates()
    return df


def data_merge_all_class(data_all):
    ##用来处理分大类别计算用户价值
    a = data_all[
        ['reg_user_cnt', 'isprofit_d07_cnt', 'profit_d07', 'profit_m01', 'profit_y01', 'profit_pm08', 'profit_mm03',
         'channel_class_value_refer', 'channel_class', 'channel_department_value_refer', 'channel_department',
         'all_refer', 'all_compute', 'channel_category_value_refer', 'channel_category', 'type', 'ds']].groupby(
        ['channel_class_value_refer', 'channel_category_value_refer', 'channel_category',
         'channel_department_value_refer', 'channel_department', 'all_refer', 'all_compute', 'type', 'ds']).sum()
    a = a.reset_index()
    a[ 'channel_class'] = a['channel_class_value_refer']
    a = a.reindex_axis(['channel_class'] + list(a.columns[:-1]), axis=1)

    b = data_all[
        ['reg_user_cnt', 'isprofit_d07_cnt', 'profit_d07', 'profit_m01', 'profit_y01', 'profit_pm08', 'profit_mm03',
         'channel_class_value_refer', 'channel_class', 'channel_department_value_refer', 'channel_department',
         'all_refer', 'all_compute', 'channel_category_value_refer', 'channel_category', 'type', 'ds']].groupby(
        ['channel_category_value_refer', 'channel_category', 'channel_department_value_refer', 'channel_department',
         'all_refer', 'all_compute', 'type', 'ds']).sum()
    b = b.reset_index()
    b[['channel_class_value_refer', 'channel_class']] = b[['channel_category_value_refer', 'channel_category']]
    b = b.reindex_axis(['channel_class_value_refer', 'channel_class'] + list(b.columns[:-2]), axis=1)

    c = data_all[
        ['reg_user_cnt', 'isprofit_d07_cnt', 'profit_d07', 'profit_m01', 'profit_y01', 'profit_pm08', 'profit_mm03',
         'channel_class_value_refer', 'channel_class', 'channel_department_value_refer', 'channel_department',
         'all_refer', 'all_compute', 'channel_category_value_refer', 'channel_category', 'type', 'ds']].groupby(
        ['channel_department_value_refer', 'channel_department', 'all_refer', 'all_compute', 'type', 'ds']).sum()
    c = c.reset_index()
    c[['channel_class_value_refer', 'channel_class']] = c[['channel_department_value_refer', 'channel_department']]
    c[['channel_category_value_refer', 'channel_category']] = c[
        ['channel_department_value_refer', 'channel_department']]
    c = c.reindex_axis(
        ['channel_class_value_refer', 'channel_class', 'channel_category_value_refer', 'channel_category'] + list(
            c.columns[:-4]), axis=1)

    d = data_all[
        ['reg_user_cnt', 'isprofit_d07_cnt', 'profit_d07', 'profit_m01', 'profit_y01', 'profit_pm08', 'profit_mm03',
         'channel_class_value_refer', 'channel_class', 'channel_department_value_refer', 'channel_department',
         'all_refer', 'all_compute', 'channel_category_value_refer', 'channel_category', 'type', 'ds']].groupby(
        ['all_refer', 'all_compute', 'type', 'ds']).sum()
    d = d.reset_index()
    d[['channel_class_value_refer', 'channel_class']] = d[['all_refer', 'all_compute']]
    d[['channel_category_value_refer', 'channel_category']] = d[['all_refer', 'all_compute']]
    d[['channel_department_value_refer', 'channel_department']] = d[['all_refer', 'all_compute']]

    d = d.reindex_axis(
        ['channel_class_value_refer', 'channel_class', 'channel_category_value_refer', 'channel_category',
         'channel_department_value_refer', 'channel_department'] + list(
            d.columns[:-6]), axis=1)

    data = pd.concat([data_all,a
                         , b
                         , c
                         , d
                      ], ignore_index=True)
    data = data.drop_duplicates()
    return data

def usage():
    print "--channel:指定渠道，all表示所有渠道"
    print "--type:指定类型，all表示所有类型"
    print "--bdate:指定开始时间"
    print "--edate:指定结束时间"
    print "-t:指定时间（此选项不可与bdate和edate共存）"
    print "================opts like :============="
    print "python filename --channel=all --type=all --bdate=2017-07-01 --edate=2017-07-14 -t2017-07-15"

##整体计算（电商+广告毛利合并）
def get_maoli_36_month_per_rates(data_all, channel_category, ds_list):
    data = data_all[(data_all['channel_category'] == channel_category) & (data_all['ds'] >= '2015-01-01') & (
    data_all['ds'] < ds_list['ds_1y3_1'])]

    data = data.groupby(['channel_category', 'ds']).sum()
    data = data.reset_index()
    data = data.groupby(['channel_category', data['ds'].str[:7]]).sum()
    #     data=data.reset_index()
    data['m01'] = data['profit_m01'] / data['reg_user_cnt']
    data['m02'] = data['profit_m02'] / data['reg_user_cnt']
    data['m03'] = data['profit_m03'] / data['reg_user_cnt']
    data['m04'] = data['profit_m04'] / data['reg_user_cnt']
    data['m05'] = data['profit_m05'] / data['reg_user_cnt']
    data['m06'] = data['profit_m06'] / data['reg_user_cnt']
    data['m07'] = data['profit_m07'] / data['reg_user_cnt']
    data['m08'] = data['profit_m08'] / data['reg_user_cnt']
    data['m09'] = data['profit_m09'] / data['reg_user_cnt']
    data['m10'] = data['profit_m10'] / data['reg_user_cnt']
    data['m11'] = data['profit_m11'] / data['reg_user_cnt']
    data['m12'] = data['profit_m12'] / data['reg_user_cnt']
    data['m13'] = data['profit_m13'] / data['reg_user_cnt']
    data['m14'] = data['profit_m14'] / data['reg_user_cnt']
    data['m15'] = data['profit_m15'] / data['reg_user_cnt']
    data['m16'] = data['profit_m16'] / data['reg_user_cnt']
    data['m17'] = data['profit_m17'] / data['reg_user_cnt']
    data['m18'] = data['profit_m18'] / data['reg_user_cnt']
    data['m19'] = data['profit_m19'] / data['reg_user_cnt']
    data['m20'] = data['profit_m20'] / data['reg_user_cnt']
    data['m21'] = data['profit_m21'] / data['reg_user_cnt']
    data['m22'] = data['profit_m22'] / data['reg_user_cnt']
    data['m23'] = data['profit_m23'] / data['reg_user_cnt']
    data['m24'] = data['profit_m24'] / data['reg_user_cnt']
    data['m25'] = data['profit_m25'] / data['reg_user_cnt']
    data['m26'] = data['profit_m26'] / data['reg_user_cnt']
    data['m27'] = data['profit_m27'] / data['reg_user_cnt']
    data['m28'] = data['profit_m28'] / data['reg_user_cnt']
    data['m29'] = data['profit_m29'] / data['reg_user_cnt']
    data['m30'] = data['profit_m30'] / data['reg_user_cnt']
    data['m31'] = data['profit_m31'] / data['reg_user_cnt']
    data['m32'] = data['profit_m32'] / data['reg_user_cnt']
    data['m33'] = data['profit_m33'] / data['reg_user_cnt']
    data['m34'] = data['profit_m34'] / data['reg_user_cnt']
    data['m35'] = data['profit_m35'] / data['reg_user_cnt']
    data['m36'] = data['profit_m36'] / data['reg_user_cnt']
    data = data.reset_index()
    
    data = data.groupby('channel_category').mean()
    data = data.reset_index()
    data = data[
        ['m01', 'm02', 'm03', 'm04', 'm05', 'm06', 'm07', 'm08', 'm09', 'm10', 'm11', 'm12', 'm13', 'm14', 'm15', 'm16',
         'm17', 'm18', 'm19', 'm20', 'm21', 'm22', 'm23', 'm24', 'm25', 'm26', 'm27', 'm28', 'm29', 'm30', 'm31', 'm32',
         'm33', 'm34', 'm35', 'm36']]

    data_2 = data.T
    data_2 = data_2.reset_index(drop=True).reset_index()
    data_2 = data_2.rename(columns={0: 'profit'})
    data_2 = data_2.fillna(0)

    print data.head(2)
    X_train = data_2['index'].tolist()
    y_train = data_2['profit'].tolist()

    from scipy.optimize import curve_fit
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    # plt.axis([0, 20, 0, 0.2])
    # plt.plot(X_train, y_train)
    try:
        popt, pcov = curve_fit(func, X_train, y_train)
    except RuntimeError: 
        print "Error - curve_fit failed:%s" % (channel_category)

    # popt数组中，三个值分别是待求参数a,b,c
    a = popt[0]
    b = popt[1]
    c = popt[2]
    y2 = [func(i, a, b, c) for i in X_train]
    # plt.plot(X_train, y2, 'r--')

    print '%f * np.exp(-%f * x) + %f' % (a, b, c)

    # s12 = func(1, a, b, c) + func(2, a, b, c) + func(3, a, b, c) + func(4, a, b, c) + func(5, a, b,
    #                                                                                                           c) + func(
    #     6, a, b, c) + func(7, a, b, c) + func(8, a, b, c) + func(9, a, b, c) + func(10, a, b, c) + func(11, a, b, c) + func(12, a, b, c)
    s12 = data['m01'].drop_duplicates()[0]+data['m02'].drop_duplicates()[0]+data['m03'].drop_duplicates()[0]+data['m04'].drop_duplicates()[0]+data['m05'].drop_duplicates()[0]+data['m06'].drop_duplicates()[0]+data['m07'].drop_duplicates()[0]+data['m08'].drop_duplicates()[0]+data['m09'].drop_duplicates()[0]+data['m10'].drop_duplicates()[0]+data['m11'].drop_duplicates()[0]+data['m12'].drop_duplicates()[0]
    # data['r_m01'] = func(1,a,b,c)/s12
    # data['r_m02'] = func(2,a,b,c)/s12
    # data['r_m03'] = func(3,a,b,c)/s12
    # data['r_m04'] = func(4,a,b,c)/s12
    # data['r_m05'] = func(5,a,b,c)/s12
    # data['r_m06'] = func(6,a,b,c)/s12
    # data['r_m07'] = func(7,a,b,c)/s12
    # data['r_m08'] = func(8,a,b,c)/s12
    # data['r_m09'] = func(9,a,b,c)/s12
    # data['r_m10'] = func(10,a,b,c)/s12
    # data['r_m11'] = func(11,a,b,c)/s12
    # data['r_m12'] = func(12,a,b,c)/s12
    # data['r_m13'] = func(13,a,b,c)/s12
    # data['r_m14'] = func(14,a,b,c)/s12
    # data['r_m15'] = func(15,a,b,c)/s12
    # data['r_m16'] = func(16,a,b,c)/s12
    # data['r_m17'] = func(17,a,b,c)/s12
    # data['r_m18'] = func(18,a,b,c)/s12
    # data['r_m19'] = func(19,a,b,c)/s12
    # data['r_m20'] = func(20,a,b,c)/s12
    # data['r_m21'] = func(21,a,b,c)/s12
    # data['r_m22'] = func(22,a,b,c)/s12
    # data['r_m23'] = func(23,a,b,c)/s12
    # data['r_m24'] = func(24,a,b,c)/s12
    # data['r_m25'] = func(25,a,b,c)/s12
    # data['r_m26'] = func(26,a,b,c)/s12
    # data['r_m27'] = func(27,a,b,c)/s12
    # data['r_m28'] = func(28,a,b,c)/s12
    # data['r_m29'] = func(29,a,b,c)/s12
    # data['r_m30'] = func(30,a,b,c)/s12
    
    data['r_m01'] = data['m01'].drop_duplicates()[0]/s12
    data['r_m02'] = data['m02'].drop_duplicates()[0]/s12
    data['r_m03'] = data['m03'].drop_duplicates()[0]/s12
    data['r_m04'] = data['m04'].drop_duplicates()[0]/s12
    data['r_m05'] = data['m05'].drop_duplicates()[0]/s12
    data['r_m06'] = data['m06'].drop_duplicates()[0]/s12
    data['r_m07'] = data['m07'].drop_duplicates()[0]/s12
    data['r_m08'] = data['m08'].drop_duplicates()[0]/s12
    data['r_m09'] = data['m09'].drop_duplicates()[0]/s12
    data['r_m10'] = data['m10'].drop_duplicates()[0]/s12
    data['r_m11'] = data['m11'].drop_duplicates()[0]/s12
    data['r_m12'] = data['m12'].drop_duplicates()[0]/s12
    data['r_m13'] = data['m13'].drop_duplicates()[0]/s12
    data['r_m14'] = data['m14'].drop_duplicates()[0]/s12
    data['r_m15'] = data['m15'].drop_duplicates()[0]/s12
    data['r_m16'] = data['m16'].drop_duplicates()[0]/s12
    data['r_m17'] = data['m17'].drop_duplicates()[0]/s12
    data['r_m18'] = data['m18'].drop_duplicates()[0]/s12
    data['r_m19'] = data['m19'].drop_duplicates()[0]/s12
    data['r_m20'] = data['m20'].drop_duplicates()[0]/s12
    data['r_m21'] = data['m21'].drop_duplicates()[0]/s12
    data['r_m22'] = data['m22'].drop_duplicates()[0]/s12
    data['r_m23'] = data['m23'].drop_duplicates()[0]/s12
    data['r_m24'] = data['m24'].drop_duplicates()[0]/s12
    data['r_m25'] = data['m25'].drop_duplicates()[0]/s12
    data['r_m26'] = data['m26'].drop_duplicates()[0]/s12
    data['r_m27'] = data['m27'].drop_duplicates()[0]/s12
    data['r_m28'] = data['m28'].drop_duplicates()[0]/s12
    data['r_m29'] = data['m29'].drop_duplicates()[0]/s12
    data['r_m30'] = data['m30'].drop_duplicates()[0]/s12 

    data['r_m31'] = func(31,a,b,c)/s12
    data['r_m32'] = func(32,a,b,c)/s12
    data['r_m33'] = func(33,a,b,c)/s12
    data['r_m34'] = func(34,a,b,c)/s12
    data['r_m35'] = func(35,a,b,c)/s12
    data['r_m36'] = func(36,a,b,c)/s12

    data['channel_category'] = channel_category

    data = data[
        ['channel_category', 'r_m01', 'r_m02', 'r_m03', 'r_m04', 'r_m05', 'r_m06', 'r_m07', 'r_m08', 'r_m09', 'r_m10',
         'r_m11', 'r_m12', 'r_m13', 'r_m14', 'r_m15', 'r_m16', 'r_m17', 'r_m18', 'r_m19', 'r_m20', 'r_m21', 'r_m22',
         'r_m23', 'r_m24', 'r_m25', 'r_m26', 'r_m27', 'r_m28', 'r_m29', 'r_m30', 'r_m31', 'r_m32', 'r_m33', 'r_m34',
         'r_m35', 'r_m36']]
    return data

def get_result_maoli_36_month_per_rates(data_all,ds_list):
    data_list = []
    for channel_category in data_all['channel_category'].drop_duplicates():
        res = get_maoli_36_month_per_rates(data_all, channel_category, ds_list)
        data_list.append(res)
    df_total=pd.concat(data_list,ignore_index=True)
    df_total = df_total.drop_duplicates()
    return df_total

def insert_to_table_by_name(data_cur, c_path, ds,freq_type,tab_name,sysCurDate):
    # data_cur.to_csv('./user_value_auto_compute_result.csv', index=False, header=None, encoding="utf8")
    path_result = '%s/%s_%s.csv' % (c_path,tab_name,sysCurDate)
    print '生成的结果csv文件(36个月占比):',path_result
    data_cur.to_csv(path_result, index=False,header=None, float_format = '%.4f', encoding="utf8")
    cmd_hive = '''hive -e "load data local inpath \'''' + c_path + '''/%s_%s.csv\' OVERWRITE  into table %s partition(freq_type='%s',ds='%s')"''' % (tab_name,sysCurDate,tab_name,freq_type,ds)
    # 导入hive库
    print cmd_hive
    subprocess.call(cmd_hive, shell=True)
    print '生成文件:%s.csv' % (path_result)


def main():
    # 必须五个参数，否则退出
    # if len(sys.argv)!=5:
    #     sys.exit("Error input parameter , need 4 parameters......")

    # 当前日期
    sysCurDate = datetime.date.today().strftime('%Y-%m-%d')
    # 昨天--当天
    c_date = day_get(sysCurDate, 0)

    bdate = day_get(c_date, -14)
    edate = day_get(bdate, 6)

    #默认周期：按周
    freq_type='daily'

    try:
        opts, args = getopt.getopt(sys.argv[1:], "ht:w:m:", ["help", "channel=", "type=", "bdate=", "edate=", ]);

        # print("============ opts ==================");
        # print(opts);

        # print("============ args ==================");
        # print(args);

        # check all param
        for opt, value in opts:
            if opt in ("-h", "--help"):
                usage()
                sys.exit(1)
            elif opt in ("-t"):
                bdate = day_get(value, -14)
                edate = day_get(bdate, 6)
                freq_type = 'daily'
                print "按天跑，指定时间点:" + value
            elif opt in ("-m"):
                bdate = get_month_begin_and_end(get_last_month_end(value))[0]
                edate = get_month_begin_and_end(get_last_month_end(value))[1]
                freq_type = 'monthly'
                print "按月跑，指定月份:" + value
                print "取上月"+bdate+'->'+edate
            elif opt in ("-w"):
                bdate = get_firstday_of_week(get_last_lastweek_day(value))
                edate = get_lastday_of_week(get_last_lastweek_day(value))
                freq_type = 'weekly'
                print "按周跑，指定周:" + value
                print '取上上周'+bdate+'->'+edate
            elif opt == '--channel':
                print "输入渠道:" + value
                print "暂不支持输入渠道......"
            elif opt == '--type':
                print "输入类型:" + value
                print "暂不支持输入类型......"
            elif opt == '--bdate':
                bdate = value
                print "输入指定开始时间:" + value
            elif opt == '--edate':
                edate = value
                print "输入指定结束时间:" + value
            if bdate > edate:
                print("date parameter error!")
                usage()
                sys.exit(1)
    except getopt.GetoptError:
        print("getopt error!")
        usage()
        sys.exit(1)

    # 加载基础信息
    # 读取文件
    ds_list = get_date_list(bdate, edate)

    root_path = '/data1/user_value'
    data_file = '%s/user_value_auto_compute_%s.csv' % (root_path, sysCurDate)
    generate_csv_source(data_file)
    data_all = pd.read_csv(data_file)
    # 字段数据类型确定
    data_all[['profit_d07', 'profit_m01', 'profit_y01', 'profit_pm08', 'profit_mm03']] = data_all[
        ['profit_d07', 'profit_m01', 'profit_y01', 'profit_pm08', 'profit_mm03']].astype(float)

    print '计算36个月的毛利占比'
    re_36=get_result_maoli_36_month_per_rates(data_all,ds_list)
    insert_to_table_by_name(re_36, c_path, ds_list['ds'],freq_type,'rpt.full_d_mkt_user_value_36month_maoli_rate',sysCurDate)
    print '计算36个月的毛利占比--已完成'

    data_all = data_merge_all_class(data_all)

    print ds_list
    # 计算参考系数表
    coef_table_e_supplier = get_different_coef_by_class(data_all, 'e_supplier_profit', ds_list)
    coef_table_ad = get_different_coef_by_class(data_all, 'ad_profit', ds_list)
    coef_table_licai = get_different_coef_by_class(data_all, 'licai_order_profit', ds_list)

    coef_table_e_supplier=optimized_coef_table(coef_table_e_supplier)
    coef_table_ad = optimized_coef_table(coef_table_ad)
    coef_table_licai = optimized_coef_table(coef_table_licai)

    coef_table_e_supplier.to_csv('%s/coef_table_e_supplier.csv' % (root_path), index=False, encoding="gb18030")
    coef_table_ad.to_csv('%s/coef_table_ad.csv' % (root_path), index=False, encoding="gb18030")
    coef_table_licai.to_csv('%s/coef_table_licai.csv' % (root_path), index=False, encoding="gb18030")

    # #待计算数据
    # if len(sys.argv)==5:
    #     ##参数1表示渠道
    #     if sys.argv[1]!='auto':
    #         #无参数时，表示自动计算
    #         #计算开始日期
    #         #计算结束日期
    #         #print '自动计算所有渠道所有类型......'
    #         begin_date_0=day_get(sys.argv[3],-14)
    #         end_date_0=day_get(sys.argv[4],-7)
    #         data_1_week_ago = data_all[(data_all['ds']>begin_date_0) & (data_all['ds']<end_date_0) & (data_all['channel_class']==sys.argv[1])]
    #     else if sys.argv[1]=='auto':
    #         data_1_week_ago = data_all[(data_all['ds']>begin_date_0) & (data_all['ds']<end_date_0)]
    #     ##参数2表示类型
    #     if sys.argv[2]!='all':
    #         data_1_week_ago = data_all[(data_all['type']==sys.argv[2])]
    # else:
    #     sys.exit("Error input parameter , need 4 parameters......")

    # 计算日期：由脚本参数传入a=  datetime.datetime.strptime('2017-07-12', '%Y-%m-%d')


    data_1_week_ago = data_all[(data_all['ds'] >= ds_list['ds_0']) & (data_all['ds'] <= ds_list['ds_1'])]
    user_value_table_e_supplier = get_user_values(data_1_week_ago, coef_table_e_supplier, 'e_supplier_profit', ds_list)
    user_value_table_ad = get_user_values(data_1_week_ago, coef_table_ad, 'ad_profit', ds_list)
    user_value_table_licai = get_user_values(data_1_week_ago, coef_table_licai, 'licai_order_profit', ds_list)

    # # 合并dataframe
    # # df_total=pd.concat([coef_table,user_value_table],ignore_index=True)
    # user_value_table_e_supplier = pd.concat([user_value_table_e_supplier, coef_table_e_supplier], ignore_index=True)
    # user_value_table_ad = pd.concat([user_value_table_ad, coef_table_ad], ignore_index=True)
    # user_value_table_licai = pd.concat([user_value_table_licai, coef_table_licai], ignore_index=True)

    #
    a = pd.merge(user_value_table_e_supplier, user_value_table_ad,
                 on=['column_name', 'channel', 'ds', 'channel_category', 'channel_department'], how='left',
                 suffixes=('_supplier', '_ad'))
    b = pd.merge(a, user_value_table_licai,
                 on=['column_name', 'channel', 'ds', 'channel_category', 'channel_department'], how='left',
                 suffixes=('', '_licai'))

    result = b.loc[:, ['ds', 'column_name', 'channel', 'channel_category', 'channel_department', 'type_name_supplier',
                       'reg_user_cnt_supplier', 'rate_d07',
                       'user_value_supplier', 'm_w_ds_supplier', \
                       'm_w_coef', 'm_w_intercept', 'y_m_ds_supplier', 'y_m_multiple_supplier', \
                       'type_name', 'profit_d07_licai', 'user_value', 'm_w_ds', 'm_w_multiple_licai', 'y_m_ds',
                       'y_m_multiple', \
                       'type_name_ad', 'profit_d07', 'user_value_ad', 'm_w_ds_ad', 'm_w_multiple', 'y_m_ds_ad',
                       'y_m_multiple_ad' \
                       ]]
    result['user_value_all'] = result['user_value_supplier'] + result['user_value'] + result['user_value_ad']

    result = result.loc[:,
             ['channel', 'channel_category', 'channel_department', 'reg_user_cnt_supplier',
              'user_value_all', 'type_name_supplier', 'rate_d07',
              'user_value_supplier', 'm_w_ds_supplier', \
              'm_w_coef', 'm_w_intercept', 'y_m_ds_supplier', 'y_m_multiple_supplier', \
              'type_name', 'profit_d07_licai', 'user_value', 'm_w_ds', 'm_w_multiple_licai', 'y_m_ds', 'y_m_multiple', \
              'type_name_ad', 'profit_d07', 'user_value_ad', 'm_w_ds_ad', 'm_w_multiple', 'y_m_ds_ad',
              'y_m_multiple_ad', 'ds' \
              ]]

    # 有中文的字段的转码
    result['channel'] = result['channel'].apply(lambda a: a.decode('utf8'))
    result['channel_category'] = result['channel_category'].apply(lambda a: a.decode('utf8'))
    result['channel_department'] = result['channel_department'].apply(lambda a: a.decode('utf8'))

    # 存进sqlserver
    # get_sqlserver_table(result,a_period_0)
    insert_to_table(result, root_path, ds_list['ds'],freq_type)

    print result.columns
    print "end compute......"

    


def test_func():
    data_all = pd.read_csv('./user_value_auto_compute.csv')
    # 字段数据类型确定
    data_all[['profit_d07', 'profit_m01', 'profit_y01', 'profit_pm08', 'profit_mm03']] = data_all[
        ['profit_d07', 'profit_m01', 'profit_y01', 'profit_pm08', 'profit_mm03']].astype(float)

    # 当前日期
    sysCurDate = datetime.date.today().strftime('%Y-%m-%d')
    # 昨天
    c_date = day_get(sysCurDate, -1)

    bdate = day_get(c_date, -14)
    edate = day_get(c_date, -7)

    ds_list = get_date_list(bdate, edate)

    data_all = data_all[(data_all['channel_department'] == '效果投放') & (data_all['type'] == 'ad_profit')]

    # 计算参考系数表
    # coef_table_e_supplier = get_different_coef_by_class(data_all, 'e_supplier_profit', ds_list)
    coef_table_ad = get_different_coef_by_class(data_all, 'ad_profit', ds_list)
    # coef_table_licai = get_different_coef_by_class(data_all, 'licai_order_profit', ds_list)

    print coef_table_ad.head()


if __name__ == '__main__':
    main()

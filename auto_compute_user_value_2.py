#!/usr/bin/env python
# coding=UTF-8

## ======================================================
## @ScriptName:      user_value_auto_compute_2.py
## @Author:          xiangyu.xu
## @DateTime:        2017-08-31 15:52:10
## @Description:     用户价值自动化计算
## @input:           输入参数，1：渠道名称，默认为'auto',自动计算所有的渠道
##                             2: 电商类型, 默认为'all',自动计算e_supplier_profit、ad_profit、licai_order_profit三种类型
##                             3: 开始时间，默认为14天前，可以直接指定
##                             4: 结束时间，默认为7天前，可以直接指定
## ======================================================


import datetime
import calendar
import os
import getopt
import sys
import subprocess
import numpy as np
import pandas as pd
from pandas import DataFrame
from base import *
import collections

c_path = os.path.split(os.path.realpath(__file__))[0]
# c_path = '/home/xuxy/learn/user_value'

# 禁用一些警告
pd.options.mode.chained_assignment = None

def generate_csv_source(data_file):
    #判断文件是否存在
    if not os.path.exists(data_file):
        # 拉取hive表数据
        cmd_sql = 'hive -e "set hive.cli.print.header=true; \
                select * from dw.full_d_usr_channel_sum where ds<>\'\'  \
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
    path_result = '%s/user_value_auto_compute_result_v2.csv' % (c_path)
    print '生成的结果csv文件:',path_result
    data_cur.to_csv(path_result, index=False,header=None, float_format = '%.4f', encoding="utf8")
    cmd_hive = '''hive -e "load data local inpath \'''' + c_path + '''/user_value_auto_compute_result_v2.csv\' OVERWRITE  into table rpt.user_value_auto_compute_result_v2 partition(freq_type='%s',ds='%s')"''' % (freq_type,ds)
    # 导入hive库
    print cmd_hive
    subprocess.call(cmd_hive, shell=True)
    print '生成文件:user_value_auto_compute_result.csv'


def add_columns(data):
    # 加字段：七天购物率，人均月毛利，人均年毛利
    data['rate_d07'] = data['isprofit_d07_cnt'] / data['reg_user_cnt']
    data['profit_m01_avg'] = data['profit_m01'] / data['reg_user_cnt']
    data['profit_y01_avg'] = data['profit_y01'] / data['reg_user_cnt']
    return data

def format_col(data):
    # 有中文的字段的转码
    data['ora_source'] = data['ora_source'].apply(lambda a: str(a).decode('utf8'))
    return data


def delete_exception_value(df):
    ##根据人均首月毛利去除异常值，3倍标准差之外的去掉，注册数小于50的去掉
    ##去掉每天注册数低于50的数据
    data = df[(np.abs((df['profit_m01'] / df['reg_user_cnt']) - (
        df['profit_m01'] / df['reg_user_cnt']).std()) <= 3 * np.mean(
        df['profit_m01'] / df['reg_user_cnt'])) & (df['reg_user_cnt'] > 0) & (df['reg_user_cnt']>=50)]
    # data.to_csv('./data_1234.csv', index=False, encoding="gb18030")
    return data

# 计算系数，包括月周比（斜率a、截距b）、年月比
# 计算电商的系数(已到最小的需要计算的类别了)
def get_coef_of_e_supplier(data, ds_list):
    # 获取首月和七天购物率的关系
    # 首月(T-90至T-180的注册用户)
    data_1_month_ago = data[(data['ds'] >= ds_list['ds_3m_0']) & (data['ds'] <= ds_list['ds_3m_1'])]
    if data_1_month_ago.empty:
        a=0
        b=0
        print "Empty data:"
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
    #先校验
    if (a!=0 or b!=0) and x>=1:
        valid=1
    else :
        valid=0
    return {"ds": ds_list['ds'],
            "reg_user_cnt": reg_user_cnt,
            "rate_d07": rate_d07,
            "user_value": v,
            "m_w_ds": ds_list['ds_3m'],
            "m_w_coef": a,
            "m_w_intercept": b,
            "y_m_ds": ds_list['ds_1y'],
            "y_m_multiple": x,
            "valid":valid
            }


# 计算广告、理财的系数
def get_coef_of_ad_or_licai(data, ds_list):
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

    # 先校验
    if x1 >= 1 and x2 >= 1:
        valid = 1
    else:
        valid = 0
    rdict = collections.OrderedDict()
    rdict = {"ds": ds_list['ds'],
            "reg_user_cnt": reg_user_cnt,
            "profit_d07": profit_d07,
            "user_value": v,
            "m_w_ds": ds_list['ds_1m'],
            "m_w_multiple": x1,
            "y_m_ds": ds_list['ds_1y'],
            "y_m_multiple": x2,
            "valid":valid
            }
    return rdict

# 循环获取维度列，生成一个一个带计算系数的dataframe：
# 包含字段：['type','column_name','channel','ds','reg_user_cnt','rate_d07','user_value','m_w_ds','m_w_coef','m_w_intercept','y_m_ds','y_m_multiple']
# 定义：     计算时间段、注册数、七天购物率、用户价值、月周比参考时间段、月周比系数、月周比截距、年月比参考时间段、年月比倍数
def get_different_coef_by_class(data_all, type_name, ds_list):

    # 定义参考的维度列，后期用户价值需要根据此计算
    cols = ['ora_source',
           'ora_channel',
           'ora_activity_type',
           'ora_parent_channel',
           'ora_channel_type',
           'ora_business_type']
    xishu_table = []
    colList = get_descending_list_order(cols)
    # colList = [['ora_business_type']]
    # print colList
    for col in colList:

        # print col
        data = data_all[data_all['type']==type_name].groupby(col+['ds']).sum()
        data = data.reset_index()
        data = add_columns(data)

        # #测试
        # if 'ora_channel' not in col:
        #     continue
        # print data.head(1)

        for index1, row in data[(data['ds'] >= ds_list['ds_0']) & (data['ds'] <= ds_list['ds_1']) ][col].drop_duplicates().iterrows():
            # print row
            contition = gen_condition_str_by_series(row).decode('utf8')
            # print contition
            # print a

            head_col = collections.OrderedDict()

            head_col = collections.OrderedDict.fromkeys(cols)
            for i in head_col:
                if i in col:
                    head_col[i] = row[i]
                    # data[i] = data[i].apply(lambda a: str(a).decode('utf8'))
                else:
                    head_col[i] = '-'
            # print head_col
            data1=data[eval(contition)]
            data1 = delete_exception_value(data1)
            if data1.empty:
                continue
            if type_name in ['e_supplier_profit', 'common_order_profit', 'tao2_order_profit', 'zy_order_profit']:
                coef_dict = get_coef_of_e_supplier(data1, ds_list)
            if type_name in ['ad_profit', 'ad_common_profit', 'ad_licai_profit', 'licai_order_profit']:
                coef_dict = get_coef_of_ad_or_licai(data1, ds_list)
            # print coef_dict

            coef_dict = collections.OrderedDict(head_col.items() + coef_dict.items())
            xishu_table.append(coef_dict)
            # print xishu_table
            # break

    df = DataFrame(xishu_table)
    df = df.drop_duplicates()
    return df

# ,这里只计算一个类别
def compute_user_value(coef_table, row, type_name, ds_list):
    # 定义参考的维度列，后期用户价值需要根据此计算
    cols = ['ora_source',
            'ora_channel',
            'ora_activity_type',
            'ora_parent_channel',
            'ora_channel_type',
            'ora_business_type']
    xishu_table = []
    colList = get_ascending_list_order(cols)

    reg_user_cnt = row['reg_user_cnt'][0]
    isprofit_d07_cnt = row['isprofit_d07_cnt'][0]
    profit_d07 = row['profit_d07'][0]
    if reg_user_cnt > 0:
        rate_d07 = isprofit_d07_cnt * 1.0 / reg_user_cnt
    else:
        rate_d07 = 0


    ref_col = []
    ##如果小类没有则找上一级，根据valid=1标识
    for c in colList:
        k = {
            'ora_source':'-',
            'ora_channel':'-',
            'ora_activity_type':'-',
            'ora_parent_channel':'-',
            'ora_channel_type':'-',
            'ora_business_type':'-'
        }
        for i in c:
            k[i] = row[i][0].decode('utf8')
        s = dict_to_contition(k)
        ref_col = dict_to_contition2(k)
        dd = coef_table[(coef_table['valid'] == 1) & eval(s)]
        if not dd.empty:
            break

    # 用户价值：v=(a*rate_d07+b)*x
    if reg_user_cnt == 0:
        v = 0
    if dd.empty:
        v=0
        ref_dict={
            "m_w_ds": [''],
            "ds": [ds_list['ds']],
            "y_m_ds":[''],
            "m_w_coef": [0],
            "rate_d07": [0],
            "user_value": [0],
            "m_w_intercept": [0],
            "reg_user_cnt": [0],
            "y_m_multiple": [0],
            "y_m_ds": [0],
            "m_w_multiple": [0],
            "ref_col":['']
        }
    else:
        ref_dict=dd.to_dict(orient='list')
    if type_name in ['e_supplier_profit', 'common_order_profit', 'tao2_order_profit', 'zy_order_profit']:
        # v = (a*rate_d07+b)*x*1.0
        v = (ref_dict['m_w_coef'][0] * rate_d07 + ref_dict['m_w_intercept'][0]) * ref_dict['y_m_multiple'][0] * 1.0
        if v<0:
            v=0
        return_dict = {"ds": ds_list['ds'],
                       "reg_user_cnt": reg_user_cnt,
                       "rate_d07": rate_d07,
                       "user_value": v,
                       "m_w_ds": ref_dict['m_w_ds'][0],
                       "m_w_coef": ref_dict['m_w_coef'][0],
                       "m_w_intercept": ref_dict['m_w_intercept'][0],
                       "y_m_ds": ref_dict['y_m_ds'][0],
                       "y_m_multiple": ref_dict['y_m_multiple'][0],
                       "ref_col":ref_col
                       }
    if type_name in ['ad_profit', 'ad_common_profit', 'ad_licai_profit', 'licai_order_profit']:
        v = (profit_d07 * ref_dict['m_w_multiple'][0]) * ref_dict['y_m_multiple'][0] * 1.0 / reg_user_cnt
        return_dict = {"ds": ds_list['ds'],
                       "reg_user_cnt": reg_user_cnt,
                       "profit_d07": profit_d07,
                       "user_value": v,
                       "m_w_ds": ref_dict['m_w_ds'][0],
                       "m_w_multiple": ref_dict['m_w_multiple'][0],
                       "y_m_ds": ref_dict['y_m_ds'][0],
                       "y_m_multiple": ref_dict['y_m_multiple'][0],
                       "ref_col":ref_col
                       }
    # return type_name,a_period_0,reg_user_cnt,rate_d07,v,m_w_ds,a,b,y_m_ds,x
    # 重构ref_dict，作为返回
    return return_dict

def get_user_values(coef_table,data_all, type_name, ds_list):

    # 定义参考的维度列，后期用户价值需要根据此计算
    cols = ['ora_source',
           'ora_channel',
           'ora_activity_type',
           'ora_parent_channel',
           'ora_channel_type',
           'ora_business_type']
    xishu_table = []
    data = data_all[(data_all['type'] == type_name) & (data_all['ds'] >= ds_list['ds_0']) & (data_all['ds'] <= ds_list['ds_1'])].groupby(cols).sum()
    data = data.reset_index()
    data = add_columns(data)
    for index1, row in data[cols].drop_duplicates().iterrows():
        # print row
        contition = gen_condition_str_by_series(row).decode('utf8')
        # print contition
        # print a

        head_col = collections.OrderedDict()

        head_col = collections.OrderedDict.fromkeys(cols)
        for i in head_col:
            if i in cols:
                head_col[i] = row[i]
            else:
                head_col[i] = '-'
        # print head_col

        r = data[eval(contition)].to_dict(orient='list')

        if data.empty:
            continue
        coef_dict = compute_user_value(coef_table, r, type_name, ds_list)
        # print coef_dict

        coef_dict = collections.OrderedDict(head_col.items() + coef_dict.items())

        xishu_table.append(coef_dict)
    df = DataFrame(xishu_table)
    return df

def usage():
    print "--channel:指定渠道，all表示所有渠道"
    print "--type:指定类型，all表示所有类型"
    print "--bdate:指定开始时间"
    print "--edate:指定结束时间"
    print "-t:指定时间（此选项不可与bdate和edate共存）"
    print "================opts like :============="
    print "python filename --channel=all --type=all --bdate=2017-07-01 --edate=2017-07-14 -t2017-07-15"

def main():
    # 必须五个参数，否则退出
    # if len(sys.argv)!=5:
    #     sys.exit("Error input parameter , need 4 parameters......")

    # 当前日期
    sysCurDate = datetime.date.today().strftime('%Y-%m-%d')
    # 昨天
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
    root_path = '/data1/user_value/version2'
    data_file = '%s/full_d_usr_channel_sum%s.csv' % (root_path, sysCurDate)
    generate_csv_source(data_file)
    data_all = pd.read_csv(data_file)

    # 字段数据类型确定
    data_all[['profit_d07', 'profit_m01', 'profit_y01', 'profit_pm08', 'profit_mm03']] = data_all[
        ['profit_d07', 'profit_m01', 'profit_y01', 'profit_pm08', 'profit_mm03']].astype(float)

    # data_all = data_merge_all_class(data_all)
    print data_all.columns
    ds_list = get_date_list(bdate, edate)
    # print ds_list
    # 计算参考系数表
    coef_table_e_supplier = get_different_coef_by_class(data_all, 'e_supplier_profit', ds_list)
    coef_table_ad = get_different_coef_by_class(data_all, 'ad_profit', ds_list)
    coef_table_licai = get_different_coef_by_class(data_all, 'licai_order_profit', ds_list)

    e_supplier = get_user_values(coef_table_e_supplier, data_all, 'e_supplier_profit', ds_list)
    ad = get_user_values(coef_table_ad, data_all, 'ad_profit', ds_list)
    licai = get_user_values(coef_table_licai, data_all, 'licai_order_profit', ds_list)

    # coef_table_e_supplier.to_csv('./coef_table_e_supplier.csv', index=False, encoding="gb18030")
    # coef_table_ad.to_csv('./coef_table_ad.csv', index=False, encoding="gb18030")
    # coef_table_licai.to_csv('./coef_table_licai.csv', index=False, encoding="gb18030")

    # e_supplier.to_csv('./e_supplier.csv', index=False, encoding="gb18030")
    # ad.to_csv('./ad.csv', index=False, encoding="gb18030")
    # licai.to_csv('./licai.csv', index=False, encoding="gb18030")

    #
    a = pd.merge(e_supplier, ad,
                 on=['ora_source','ora_channel','ora_activity_type','ora_parent_channel','ora_channel_type','ora_business_type','ds'], how='left',
                 suffixes=('_supplier', '_ad'))
    b = pd.merge(a, licai,
                 on=['ora_source','ora_channel','ora_activity_type','ora_parent_channel','ora_channel_type','ora_business_type','ds'], how='left',
                 suffixes=('', '_licai'))

    result = b.loc[:, ['ds', 'ora_source','ora_channel','ora_activity_type','ora_parent_channel','ora_channel_type','ora_business_type', 'type_name_supplier',
                       'reg_user_cnt_supplier', 'rate_d07',
                       'user_value_supplier', 'm_w_ds_supplier', \
                       'm_w_coef', 'm_w_intercept', 'y_m_ds_supplier', 'y_m_multiple_supplier','ref_col_supplier', \
                       'type_name', 'profit_d07_licai', 'user_value', 'm_w_ds', 'm_w_multiple_licai', 'y_m_ds',
                       'y_m_multiple','ref_col',\
                       'type_name_ad', 'profit_d07', 'user_value_ad', 'm_w_ds_ad', 'm_w_multiple', 'y_m_ds_ad',
                       'y_m_multiple_ad' , 'ref_col_ad'\
                       ]]
    result['user_value_all'] = result['user_value_supplier'] + result['user_value'] + result['user_value_ad']

    result = result.loc[:,
             ['ora_source','ora_channel','ora_activity_type','ora_parent_channel','ora_channel_type','ora_business_type', 'reg_user_cnt_supplier',
              'user_value_all', 'type_name_supplier', 'rate_d07',
              'user_value_supplier', 'm_w_ds_supplier', \
              'm_w_coef', 'm_w_intercept', 'y_m_ds_supplier', 'y_m_multiple_supplier', 'ref_col_supplier',\
              'type_name', 'profit_d07_licai', 'user_value', 'm_w_ds', 'm_w_multiple_licai', 'y_m_ds', 'y_m_multiple','ref_col', \
              'type_name_ad', 'profit_d07', 'user_value_ad', 'm_w_ds_ad', 'm_w_multiple', 'y_m_ds_ad',
              'y_m_multiple_ad', 'ref_col_ad','ds' \
              ]]

    # 有中文的字段的转码
    # result['ora_source'] = result['ora_source'].apply(lambda a: a.decode('utf8'))
    # result['ora_channel'] = result['ora_channel'].apply(lambda a: a.decode('utf8'))
    # result['ora_activity_type'] = result['ora_activity_type'].apply(lambda a: a.decode('utf8'))
    # result['ora_parent_channel'] = result['ora_parent_channel'].apply(lambda a: a.decode('utf8'))
    # result['ora_channel_type'] = result['ora_channel_type'].apply(lambda a: a.decode('utf8'))
    # result['ora_business_type'] = result['ora_business_type'].apply(lambda a: a.decode('utf8'))
    #
    # result['ref_col_supplier'] = result['ref_col_supplier'].apply(lambda a: a.decode('utf8'))
    # result['ref_col_ad'] = result['ref_col_ad'].apply(lambda a: a.decode('utf8'))
    # result['ref_col'] = result['ref_col'].apply(lambda a: a.decode('utf8'))

    # result.to_csv('./result.csv', index=False, encoding="gb18030")

    # 存进sqlserver
    # get_sqlserver_table(result,a_period_0)
    insert_to_table(result, root_path, ds_list['ds'], freq_type)


    print "end compute......"


if __name__ == '__main__':
    main()
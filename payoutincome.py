#!/usr/bin/env python
# coding=UTF-8

import datetime
import calendar
import os
import getopt
import sys
import subprocess
import numpy as np
import pandas as pd
from pandas import DataFrame
#from base import *
import collections

import sqlite3

income_df = pd.read_csv("income.csv")
payout_df = pd.read_csv("payout.csv")
#print(income_df)
#income_df.sort_index(axis=0)
#print(income_df.sort_values(by = 'id',axis = 0,ascending = True))
data_user = income_df.drop_duplicates(['user_id'])['user_id']
#data = data.reset_index()

for i in data_user:
    data_payout = payout_df[payout_df['user_id'] == i]
    data_income = income_df[income_df['user_id'] == i]

    payout_seq = data_payout['seq']

    for j in payout_seq:



    print(payout_seq)



#payout_df = payout_df.sort_values(by = 'id',axis = 0,ascending = True)
#print(payout_df)
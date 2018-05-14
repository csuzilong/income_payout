#!/usr/bin/env python
# coding=UTF-8

import paramiko
import sys


def sftp_stor_files(local_path, remote_path, sftp_server, sftp_user, sftp_password):
    t = paramiko.Transport((sftp_server, 22))
    t.connect(username=sftp_user, password=sftp_password, hostkey=None)
    sftp = paramiko.SFTPClient.from_transport(t)

    sftp.put(local_path, remote_path)
    # sftp.remove('/home/biao.liu/fanlicard/getdata')

    t.close()


sftp_server = '192.168.3.170'
sftp_user = 'biao.liu'
sftp_password = 'biao.liu@2017'
file_n = 'asd.csv'
local_path = 'C:\\Users\\biao.liu\\Desktop\\fanlidata\\' + file_n
remote_path = '/home/biao.liu/fanlicard/getdata/' +file_n
print(local_path)
print(remote_path)
sftp_stor_files(local_path, remote_path, sftp_server, sftp_user, sftp_password)

'''
list = ['20180104','20180105','20180106','20180107','20180108','20180109','20180110','20180111']
for i in list:
    file_n = 'FLW_USER_LEVEL' + i +'.csv'
    local_path = '/home/biao.liu/fanlicard/getdata/' + file_n
    remote_path = '/Fanli_Daily/' +file_n
    sftp_stor_files(local_path, remote_path, sftp_server, sftp_user, sftp_password)
'''

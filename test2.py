#!/usr/bin/env python
# coding=UTF-8

from ftplib import FTP

def ftpconnect(host, username, password):
    ftp = FTP()
    ftp.set_debuglevel(2)
    ftp.connect(host,22)
    ftp.login(username, password)
    print ftp.getwelcome()
    return ftp

#从ftp下载文件
def downloadfile(ftp, remotepath, localpath):
    bufsize = 1024
    fp = open(localpath, 'wb')
    ftp.retrbinary(remotepath, fp.write, bufsize)
    ftp.set_debuglevel(0)
    fp.close()

if __name__ == "__main__":
    ftp = ftpconnect("192.168.3.170", "biao.liu", "biao.liu@2017")
    #downloadfile(ftp, "/Fanli_Daily/FLW_USER_LEVEL20180104.csv", "/home/biao.liu/fanlicard/getdata/FLW_USER_LEVEL20180104.csv")
    ftp.quit()

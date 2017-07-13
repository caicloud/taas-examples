# -*- coding: utf-8 -*-
"""
Created on Fri Jun 02 12:03:44 2017

@author: hyy
"""

import MySQLdb
# import sys
import pandas as pd 
import numpy as np 
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import time
import os


N = 1  # 每次从数据库中只取出一行进行文本输出展现
Fname = 'replace.csv'
PREDICT_TXT = '10002.txt'  #sys.argv[2]  # '../result/predict.txt'
CHAR_COUNT = 20
SHUCHU_TXT = 'shuchu10002.txt'


def read_dict(Fname):
    data_dict = {}
    with open(Fname) as fp:
        for line in fp:
            origin, replace = line.strip().split(',')
            times = origin.split(":")
            buff = ""
            for time in times:
                buff += time + ":"
            buff = buff.strip(":")
            data_dict[buff] = replace
    return data_dict
 


def read_mysql():
    conn = MySQLdb.connect(host='42.159.115.103', port=3306, user='root', passwd='caicloud', db='yuhuangshan', charset='utf8')
    select_result = "select * from yuhuangshan.blackmetal_result order by id desc limit %d;" %N
    monitor_previous = "select * from yuhuangshan.blackmetal_monitor order by id desc limit %d;" %N
    try:
        conn.ping()
    except Exception,e:
        print "read_mysql, Msql出了问题"
        print str(e)
        while True: 
            try: 
                conn = MySQLdb.connect(host='42.159.115.103', port=3306, user='root', passwd='caicloud', db='yuhuangshan', charset='utf8')
                print "Msql连接"
                break
            except Exception,e:
                print "尝试重连接失败" 
                time.sleep(2)
                continue
    cur = conn.cursor()
    try:
        cur.execute(select_result)
        conn.commit()
        cds = cur.fetchall()
        data = pd.DataFrame(list(cds))
        data = data.iloc[:, 1:] 
        if (data.iloc[0, 2]==1 and data.iloc[0, 3]==1):
            data.iloc[0, 2] = 0
        elif (data.iloc[0, 2]==0 and data.iloc[0, 3]==1):
            data.iloc[0, 2] = 2
        else:
            pass
        cur.execute(monitor_previous)
        data_previous = cur.fetchall()
        previous = pd.DataFrame(list(data_previous))
        previous = previous.iloc[:, 1:4]
        if previous.values.tolist() != data.iloc[:, :3].values.tolist() :
            monitor_insert = "INSERT INTO yuhuangshan.blackmetal_monitor(date_dt,time,label) values (\"%s\", \"%s\", %s)" % (data.iloc[0, 0], data.iloc[0, 1], data.iloc[0, 2])
            cur.execute(monitor_insert)
            conn.commit()
            print "insert blackmetal_monitor success" 
    except Exception,e: 
        print str(e)
    cur.close()
    conn.close() 
    return data.iloc[:, :3]



def read_mysql_batch(batch_size=300):
    conn = MySQLdb.connect(host='42.159.115.103', port=3306, user='root', passwd='caicloud', db='yuhuangshan', charset='utf8',)
    if batch_size < 500:
        sql = "select * from yuhuangshan.blackmetal_result order by id desc limit %d;" % batch_size
    else:
        sql = "select * from yuhuangshan.blackmetal_result order by id desc limit 500;" 
    try:
        conn.ping()
    except Exception,e:
        print "read_mysql_batch, Msql出了问题"
        print str(e)
        while True: 
            try: 
                conn = MySQLdb.connect(host='42.159.115.103', port=3306, user='root', passwd='caicloud', db='yuhuangshan', charset='utf8')
                print "Msql连接"
                break
            except Exception,e:
                print "尝试重连接失败" 
                time.sleep(2)
                continue
    cur = conn.cursor()
    try:  
        cur.execute(sql)
        conn.commit()
        cds = cur.fetchall()
        data = pd.DataFrame(list(cds))
        data = data.iloc[:,1:]
        for i in range(0, batch_size):
            if (data.iloc[i, 2]==1 and data.iloc[i, 3]==1):
                data.iloc[i, 2] = 0
            elif (data.iloc[i, 2]==0 and data.iloc[i, 3]==1):
                data.iloc[i, 2] = 2
            else:
                pass
    except Exception,e:
        print str(e)
    cur.close()
    conn.close()       
    return data.iloc[:, :3]



def read_lastline():
    line = []
    try:
        with open(PREDICT_TXT) as fp:
            fp.seek(-CHAR_COUNT, 2)
            lines = fp.readlines()
            line = lines[-1]
    except IOError:
        print "Error: 读取文件失败"
        with open(PREDICT_TXT, 'w') as fp:
            line = '[myscetion]'
            fp.write(line)
            fp.write('\n')
    else:
        print "读取文件成功"       
    return line.strip().split(',')


def mail_monitor(content, title):
    receivers = ['892172919@qq.com', 'hyy@caicloud.io', 'huihui@caicloud.io', 'dagan3200@126.com']  
    message = MIMEText(content,'plain','utf-8')  # 发送内容

    sender = 'key<892172919@qq.com>'
    message['From'] = 'key<892172919@qq.com>'
    message['To'] = "892172919@qq.com, hyy@caicloud.io, huihui@caicloud.io, dagan3200@126.com"
    message['Subject'] = title  
    try:
        smtpObj = smtplib.SMTP_SSL('smtp.qq.com')
        smtpObj.login('892172919', 'xvbqfpnmihibbcgd')
        smtpObj.sendmail(sender, receivers, message.as_string())
        smtpObj.quit()
        print('mail send success')
    except smtplib.SMTPException as e:
        print('error',e)  #打印错误       







def main():
    content = '\n结果输出程序已启动，数据将输出到10002.txt文档。\n如需确认，请查看文档。\n\n\n如有疑问，请咨询:18662581318'
    title = '10002输出启动：结果输出程序已启动'
    mail_monitor(content, title)
    data_dict = read_dict(Fname)
    line_count = 500
    try:
        with open(SHUCHU_TXT, 'r') as of:
            line_count = len(of.readlines())
    except:
        print SHUCHU_TXT+'文件不存在'
    # 行数监控
    if line_count > 500:
        line_count = 500
    pre_n = read_mysql_batch(line_count)  # 预测表中读取n行
    for i in range(len(pre_n[3])):    # 正则化最后一列为 =label
        pre_n.iloc[i,-1] = '=' + str(pre_n.iloc[i,-1])
    time = pre_n.iloc[:,1].values
    for i in range(len(pre_n)):  # 时间匹配
        pre_n.iloc[i,1] = data_dict[time[i]][1:]

    if os.path.exists(PREDICT_TXT):  #设置表头
        fw = open(PREDICT_TXT,"r+")
        first_line = fw.read(11)
        if first_line != '[myscetion]':
            fw.seek(0)
            fw.write('[myscetion]' + "\n")
    else:
        fw = open(PREDICT_TXT,"w")
        fw.write('[myscetion]' + "\n")
    
    for i in range(0, len(pre_n)):  # 写入n行预测数据
        output = "".join(map(str, pre_n.iloc[i, :]))
        fw.seek(0, 2)
        fw.write(output + "\n")
    fw.close()
    print 'first to txt done'
    
    while (1):
        pre_data = read_mysql().values.tolist()[0]
        re = '=' + str(int(float(pre_data[2])))
        time = pre_data[1].strip()
        # time = '9:30:00'
        try:
            pre_data[1] = data_dict[time][1:]
            re_data = "".join(pre_data[0:2])+re
            print "re_data is :",re_data
            line = read_lastline()[0]
            print "line is :", line
            if line == re_data:
                continue
            else:
                with open(PREDICT_TXT, "a") as fw:
                    fw.write(re_data + "\n")           
        except KeyError:
            print "KeyError, time mismatch"
        else:
             print 'pre_data is writing to txt'




if __name__ == "__main__":
    main()



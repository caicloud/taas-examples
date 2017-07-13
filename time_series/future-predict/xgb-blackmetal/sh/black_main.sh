#!/usr/bin/env bash
cd `dirname $0`/..
nohup python script/load_model.py model/xgb_long.model model/xgb_short.model &>> log/load_model.log &
nohup python script/monitor_read.py &>> log/monitor_read.log &
nohup python script/monitor_write.py &>> log/monitor_write.log &


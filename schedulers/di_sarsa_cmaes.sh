#!/bin/bash
stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 2 --defense-agents 1 \
--port 8060 --deterministic --headless --hfo-logging --trials 5200000 --seed 1 > logs/di_sarsa_cmaes.log 2>&1 &


PID=$!
cd di_sarsa_cmaes
sleep 5
echo $PID
python3 di_sarsa_cmaes.py --port 8060  > ../logs/di_sarsa_cmaes_debug.log  2> ../logs/di_sarsa_cmaes_debug_error.log 
# kill $PID
# killall -9 rcssserver
kill $PID
sleep 5
cd ..

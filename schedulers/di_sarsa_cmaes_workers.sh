#!/bin/bash
port=8000
nworkers=8
episodes=300
eval_steps=25
total_steps=100
for ((p=port+10;p<=port+10*nworkers;p+=10));
do
    echo $p
    stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 2 --defense-agents 1 \
    --port $p --no-logging --headless --deterministic --trials 52000 --seed 1 > logs/di_sarsa_cmaes_workers_${p}.log 2>&1 &
done

PID=$!
cd di_sarsa_cmaes
sleep 5
echo $PID
python3 di_sarsa_cmaes_workers.py -n ${nworkers} --port_start ${port} -e ${episodes} --eval_steps ${eval_steps} --total_steps ${total_steps}
# kill $PID
# killall -9 rcssserver
kill $PID
sleep 5
cd ..

    # stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 2 --defense-agents 1 \
    # --port $p --no-logging --headless --deterministic --trials 52000 --seed 1 > logs/di_sarsa_cmaes_workers_${p}.log 2>&1 &

# stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 2 --defense-agents 1 \
# --port $p --no-logging --headless --deterministic --trials 52000 --seed 1 > logs/di_sarsa_cmaes_workers_8002.log 2>&1 &

# stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 2 --defense-agents 1 \
# --port 8003 --no-logging --headless --deterministic --trials 52000 --seed 1 > logs/di_sarsa_cmaes_workers_8003.log 2>&1 &


# stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 2 --defense-agents 1 \
# --port 800 --no-logging --headless --deterministic --trials 52000 --seed 1 > logs/di_sarsa_cmaes_workers_8001.log 2>&1 &

# PID=$!
# cd di_sarsa_cmaes
# sleep 5
# echo $PID
# python3 di_sarsa_cmaes_workers.py --port 8060 --num_worker 0 > ../logs/di_sarsa_cmaes_workers_debug.log 
# # kill $PID
# # killall -9 rcssserver
# kill $PID
# sleep 5
# cd ..

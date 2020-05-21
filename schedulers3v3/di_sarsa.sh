#!/bin/bash

loadFile=3v3_sa_weights
LearnR=0.001

stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 1 --defense-agents 2 \
--port 7160 --no-logging --no-sync --deterministic --trials 22 --seed 1 > logs/di_sarsa.log 2>&1 &

PID=$!
cd di_sarsa
sleep 5

./di_sarsa --numAgents 2 --numOpponents 3 --numEpisodes 20 --numEpisodesTest 2 --basePort 7160 \
--weightId di_sarsa_lambda_0.5_step_32_seed_1 --lambda 0.5 --step 32 --learnRate ${LearnR} --loadFile ${loadFile} --loadFile1 ${loadFile}  > ../logs/di_sarsa_debug.log

kill -SIGINT $PID
sleep 5
cd ..


stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 3 --port 7160 --no-logging --no-sync --deterministic --trials 20 --seed 1
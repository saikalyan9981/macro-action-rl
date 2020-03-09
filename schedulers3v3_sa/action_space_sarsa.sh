#!/bin/bash

stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 2 --defense-agents 1 \
--port 7120 --no-logging --headless --deterministic --trials 52000 --seed 1 > logs/action_space_sarsa.log 2>&1 &

PID=$!
cd action_space_sarsa
sleep 5

./action_space_sarsa --numAgents 1 --numOpponents 3 --numEpisodes 50000 --numEpisodesTest 2000 --basePort 7120 \
--weightId 3v3_reg_sa_action_space_sarsa_lambda_0.85_seed_1 --freq_set 8,32 --lambda 0.85 > ../logs/action_space_sarsa_debug.log

kill -SIGINT $PID
sleep 5
cd ..

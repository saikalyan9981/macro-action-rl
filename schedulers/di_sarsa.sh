#!/bin/bash

stdbuf -oL ./HFO/bin/HFO --offense-npcs 2 --defense-npcs 1 --defense-agents 1 \
--port 7160 --no-logging --no-sync --deterministic --trials 52000 --seed 1 > logs/di_sarsa.log 2>&1 &

PID=$!
cd di_sarsa
sleep 5

# ./di_sarsa --numAgents 1 --numOpponents 2 --numEpisodes 50000 --numEpisodesTest 2000 --basePort 7160 \
# --weightId di_sarsa_lambda_0.5_step_32_seed_1 --lambda 0.5 --step 32

./di_sarsa --numAgents 1 --numOpponents 2 --numEpisodes 0 --numEpisodesTest 2000 --basePort 7160 \
--weightId di_sarsa_lambda_0.5_step_32_seed_1 --lambda 0.5 --step 32 --loadFile weights_0_di_sarsa_lambda_0.5_step_32_seed_1_episode_50000
kill  $PID
echo $PID
sleep 5
cd ..

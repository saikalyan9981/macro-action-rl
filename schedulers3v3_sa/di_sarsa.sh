#!/bin/bash

Lambda=0.9
LearnR=0.0001
Step=32
Port=7160
Seed=1
TrainEpisodes=30000
TestEpisodes=2000
Eps=0.01
LoadFile=3v3_sa_weights

stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 2 --defense-agents 1 \
--port ${Port}  --no-logging --headless --deterministic --trials 42000 --seed $Seed > logs/di_sarsa.log 2>&1 &

PID=$!
echo $PID
cd di_sarsa
sleep 5

./di_sarsa --numAgents 1 --numOpponents 3 --numEpisodes ${TrainEpisodes} --numEpisodesTest ${TestEpisodes} --basePort ${Port} \
--weightId di_sarsa_lambda_${Lambda}_step_${Step}_seed_${Seed} --lambda ${Lambda} --eps ${Eps} --learnRate ${LearnR} --step ${Step}\
  \
 > ../logs/di_sarsa_debug.log

# ./di_sarsa --numAgents 1 --numOpponents 3 --numEpisodes ${TrainEpisodes} --numEpisodesTest ${TestEpisodes} --basePort ${Port} \
# --weightId di_sarsa_lambda_${Lambda}_step_${Step}_seed_${Seed} --lambda ${Lambda} --eps ${Eps} --learnRate ${LearnR} --step ${Step}\
#   > ../logs/di_sarsa_debug.log  --loadFile ${LoadFile}

kill  $PID
sleep 5
cd ..

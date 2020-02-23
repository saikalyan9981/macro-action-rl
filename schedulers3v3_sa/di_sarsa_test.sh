#!/bin/bash
for i in {5000..50000..5000}
do
    stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 2 --defense-agents 1 \
    --port 7160 --no-logging --headless --deterministic --trials 52000 --seed 1 > logs/di_sarsa.log 2>&1 &

    PID=$!
    cd di_sarsa
    sleep 5

    ./di_sarsa --numAgents 1 --numOpponents 3 --numEpisodes 0 --numEpisodesTest 2000 --basePort 7160 \
    --weightId di_sarsa_lambda_0.5_step_32_seed_1 --lambda 0.5 --step 32 --loadFile weights_0_di_sarsa_lambda_0.5_step_32_seed_1_episode_$i
    > ../logs/di_sarsa_debug.log


    kill $PID
    sleep 5
    tail -6 ../logs/di_sarsa.log > result_weights_0_di_sarsa_lambda_0.5_step_32_seed_1_episode_$i.txt

    cd ..

done

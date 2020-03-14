#!/bin/bash
Lambda=0.9
LearnR=0.01
Step=32
Port=7168
Seed=1
TrainEpisodes=0
TestEpisodes=2000
for i in {5000..50000..5000}

# for i in 50000
do

    stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 2 --defense-agents 1 \
    --port $Port --no-logging --headless --deterministic --trials 52000 --seed $Seed > logs/di_sarsa.log 2>&1 &

    # stdbuf -oL ./HFO/bin/HFO --offense-npcs 3 --defense-npcs 2 --defense-agents 1 \
    # --port $Port --no-logging --no-sync --deterministic --trials 52000 --seed $Seed > logs/di_sarsa.log 2>&1 &

    PID=$!
    cd di_sarsa
    sleep 5

    ./di_sarsa --numAgents 1 --numOpponents 3 --numEpisodes $TrainEpisodes --numEpisodesTest $TestEpisodes --basePort $Port \
    --weightId di_sarsa_lambda_${Lambda}_step_${Step}_seed_${Seed} --lambda ${Lambda} --step ${Step} --loadFile 3v3_sa_seed_${Seed}/weights_0_di_sarsa_lambda_${Lambda}_step_${Step}_seed_${Seed}_episode_$i \
    > ../logs/di_sarsa_debug.log


    kill $PID
    sleep 5
    tail -6 ../logs/di_sarsa.log > 3v3_sa_seed_${Seed}/result_weights_0_di_sarsa_lambda_${Lambda}_step_${Step}_seed_${Seed}_episode_$i.txt

    cd ..
    # break
done


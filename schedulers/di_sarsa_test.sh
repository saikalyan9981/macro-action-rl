#!/bin/bash
Step=1
Folder=2v2_step_${Step}
Port=7368
for i in {5000..50000..5000}
# for i in 50000
do
    stdbuf -oL ./HFO/bin/HFO --offense-npcs 2 --defense-npcs 1 --defense-agents 1 \
    --port $Port --no-logging --headless --deterministic --trials 52000 --seed 1 > logs/${Folder}_di_sarsa.log 2>&1 &
    # stdbuf -oL ./HFO/bin/HFO --offense-npcs 2 --defense-npcs 1 --defense-agents 1 \
    # --port $Port --no-logging --no-sync --deterministic --trials 52000 --seed 1 > logs/di_sarsa.log 2>&1 &


    PID=$!
    cd di_sarsa
    sleep 5

    ./di_sarsa --numAgents 1 --numOpponents 2 --numEpisodes 0 --numEpisodesTest 2000 --basePort $Port \
    --weightId di_sarsa_lambda_0.5_step_${Step}_seed_1 --lambda 0.5 --step ${Step} --loadFile ${Folder}/weights_0_2v2_di_sarsa_lambda_0.5_step_${Step}_seed_1_episode_$i \
    > ${Folder}_di_sarsa_debug.log


    kill $PID
    sleep 5
    tail -6 ../logs/${Folder}_di_sarsa.log > ${Folder}/result_weights_0_di_sarsa_lambda_0.5_step_${Step}_seed_1_episode_$i.txt

    cd ..
    # break

done

#  ./HFO/bin/HFO --offense-npcs 2 --defense-npcs 2 --port 7168 --no-logging --headless --deterministic --trials 2000 --seed 1 > logs/helios.log 
   
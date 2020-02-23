#code for running 3v3 HFO using helios agents.

for i in {3..5}

do
 ./HFO/bin/HFO --headless --offense-npcs=3 --defense-npcs=3 --deterministic --trials 2000 --seed $i > helios.log ; 
 tail -6 helios.log > helios-$i.txt

done
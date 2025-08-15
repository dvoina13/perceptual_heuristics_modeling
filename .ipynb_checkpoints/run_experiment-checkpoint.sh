#!/bin/bash

#  run_experiment.sh
#  Created by Doris V on 8/15/25.


#source activate pytorch

seed_arr=($(seq 1 1 10))
p_arr=(0.25 0.5 0.75)

for p in "${p_arr[@]}"
do
        for seed in "${seed_arr[@]}"
        do
            python main.py --p=$p --seed=$seed 
        done
done
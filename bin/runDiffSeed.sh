#!/bin/sh

set -xu
startSeed=$1
shift
endSeed=$1
shift
command=$1

for seed in $(seq $startSeed $endSeed)
do
    echo $seed
    $command --seed=$seed
done
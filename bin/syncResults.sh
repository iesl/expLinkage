#!/usr/bin/env bash
set -xu

dir=$1
time=$2
while true
do
rsync -avzi blake:/iesl/canvas/nishantyadav/clustering/$dir/ ../$dir/
sleep $time
done

#!/usr/bin/env bash
dir=$1
while true
do
rsync ../$dir/ -avzi blake:/iesl/canvas/nishantyadav/clustering/$dir/
sleep 60
done

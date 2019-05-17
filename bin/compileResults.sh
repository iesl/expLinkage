#!/usr/bin/env bash
#sh bin/compileResults.sh rexa 11 30 "" bestWithin_bestAcross mstWithin_bestAcross linkage_min allWithin_allAcross triplet linkage_0 linkage_max linkage_auto

set -xu

############################################### FOR COMPARING DIFFERENT OBJECTIVES ON VARYYING SEEDS ########################################################

res_root=../results_refactor

data=$1
shift

seedStart=$1
shift

seedEnd=$1
shift

suffix=$1
shift

seeds=$(seq $seedStart $seedEnd)

allObj=
while [ "$#" -gt 0 ];
do
    obj=$1
    shift
    allObj=" $allObj $obj "

    python -m utils.combineResults --outDirPrefix=BestDevThresh   --baseResDir=$res_root/d\=$data --relResultDir=BestDevThresh    --xlabel=Threshold --config=config/$data/$obj.json --seed $seeds --suffix=$suffix
#    python -m utils.combineResults --outDirPrefix=BestTestThresh  --baseResDir=$res_root/d\=$data --relResultDir=BestTestThresh   --xlabel=Threshold --config=config/$data/$obj.json --seed $seeds --suffix=$suffix

done

python -m utils.compareMethods --baseResDir=$res_root/d\=$data --outDirPrefix=BestDevThresh  --trainObj $allObj --xlabel=Threshold --seed $seeds --suffix=$suffix
#python -m utils.compareMethods --baseResDir=$res_root/d\=$data --outDirPrefix=BestTestThresh --trainObj $allObj --xlabel=Threshold --seed $seeds --suffix=$suffix

#####################################################################################################################################################################

#!/usr/bin/env bash

#set -xu

export ROOT_DIR=`pwd`
export PYTHONPATH=$ROOT_DIR/src:$PYTHONPATH
export XCLUSTER_ROOT=$ROOT_DIR/../xcluster
export XCLUSTER_JARPATH=$XCLUSTER_ROOT/target/xcluster-0.1-SNAPSHOT-jar-with-dependencies.jar

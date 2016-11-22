#!/bin/bash

TOOLBOX=./toolbox/toolbox
WORK_DIR=../data/workflow5/
DATA_FILE=../data/balken001/balken001.ply
NUM_SLICES=32

if [[ $1 == *"1"* ]] ; then
	$TOOLBOX extract-bars z -1 3 $NUM_SLICES  0.03 60 10000 0.001 $DATA_FILE $WORK_DIR/result.ply 
fi


#$TOOLBOX extract-sides 1000 0.1 0.02 400 50 0.95 0.01 ../../data/workflow1/slice_7_cluster_0.ply out2.ply


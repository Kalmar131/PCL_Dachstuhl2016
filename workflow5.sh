#!/bin/bash

TOOLBOX=./toolbox/toolbox
WORK_DIR=../data/workflow5-$1/
DATA_FILE=../data/$1/$1.ply
#DATA_FILE=Extracting/build/$1.ply
NUM_SLICES=32

rm -rf $WORK_DIR/*

#	$TOOLBOX extract-sides 1000 0.1 0.02 400 50 0.95 0.01 ../../data/workflow1/slice_7_cluster_0.ply $WORK_DIR/result2.ply

$TOOLBOX reconstruct-bars $2 z -1 3 $NUM_SLICES  0.1 60 10000 0.001 2000 0.2 0.04 $DATA_FILE $WORK_DIR/result $WORK_DIR/bar_ 


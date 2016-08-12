#!/bin/bash

TOOLBOX=./toolbox/toolbox
WORK_DIR=../data/workflow4/
DATA_FILE=../data/balken001/balken001.ply

./toolbox/toolbox statistical-outlier-removal 20 0.9 ../data/workflow1/slice_1_cluster_1.ply ../data/workflow4/tmp1.ply

./toolbox/toolbox moving-least-squares 0.01 ../data/workflow4/tmp1.ply ../data/workflow4/tmp2.ply

if [[ $1 == *"1"* ]] ; then
	$TOOLBOX estimate-normals 0.2 $DATA_FILE $WORK_DIR/result.ply
fi

if [[ $1 == *"2"* ]] ; then
	$TOOLBOX region-growing 20 50 100000 $DATA_FILE $WORK_DIR/result.ply $WORK_DIR/result2.ply
fi



./toolbox/toolbox merge-cloud ../data/workflow4/out1.ply ../data/workflow4/out9.ply ../data/workflow4/merge.ply

./toolbox/toolbox get-bounding-box ../data/workflow4/merge.ply ../data/workflow4/bbox.ply

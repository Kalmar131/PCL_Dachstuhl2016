#!/bin/bash

TOOLBOX=./toolbox/toolbox
WORK_DIR=../data/workflow2-balken18/
DATA_FILE=../data/balken18/balken18.ply
SLICE_BASE_PATTERN="slice_"
CLUSTER_BASE_PATTERN="cluster_"
BBOX_BASE_PATTERN="bbox_"
NUM_SLICES=32

rm -rf $WORK_DIR/*

if [[ $1 == *"1"* ]] ; then
	SLICE_PATTERN=$SLICE_BASE_PATTERN
	$TOOLBOX slice -1 3 $NUM_SLICES z $DATA_FILE $WORK_DIR/$SLICE_PATTERN
fi

if [[ $1 == *"2"* ]] ; then
	for SLICE_FILE in $(find $WORK_DIR -name "$SLICE_BASE_PATTERN*.ply" -printf "%f ") ; do
		CLUSTER_PATTERN=${SLICE_FILE%.ply}"_"$CLUSTER_BASE_PATTERN
		$TOOLBOX euclidian-clustering 0.1 60 10000 $WORK_DIR/$SLICE_FILE $WORK_DIR/$CLUSTER_PATTERN
	done
fi

if [[ $1 == *"3"* ]] ; then
	for CLUSTER_FILE in $(find $WORK_DIR -name "*_$CLUSTER_BASE_PATTERN*" -printf "%f ") ; do
		BBOX_FILE=${CLUSTER_FILE/$CLUSTER_BASE_PATTERN/$BBOX_BASE_PATTERN}
		$TOOLBOX get-bounding-box withEdges $WORK_DIR/$CLUSTER_FILE $WORK_DIR/$BBOX_FILE
	done
fi

if [[ $1 == *"4"* ]] ; then
	BBOX_FILES=$(find $WORK_DIR -name "*_$BBOX_BASE_PATTERN*.ply")
	$TOOLBOX merge-cloud withEdges $BBOX_FILES $WORK_DIR/result.ply
fi


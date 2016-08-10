#!/bin/bash

#./toolbox radius-outlier-removal 0.8 8 ~/Desktop/Studium/Belege/Objekterkennung/balken001/balken001.ply out.ply
#./toolbox statistical-outlier-removal 20 0.9 ~/Desktop/Studium/Belege/Objekterkennung/cluster_0.ply out.ply
#./toolbox moving-least-squares 0.02 ~/Desktop/Studium/Belege/Objekterkennung/balken001/balken001.ply out.ply

TOOLBOX=./toolbox/toolbox
BASE_DIR=../data/ScanPos/
WORK_DIR=../data/workflow2/
SCANPOS_PATTERN="ScanPos"
SLICE_BASE_PATTERN="slice_"
CLUSTER_BASE_PATTERN="cluster_"
BBOX_BASE_PATTERN="bbox_"
NUM_SLICES=6

echo $1
if [[ $1 == *"1"* ]] ; then
	for SCAN_FILE in $(find $BASE_DIR -name "$SCANPOS_PATTERN*.ply" -printf "%f ") ; do
		SLICE_PATTERN=${SCAN_FILE%.ply}"_"$SLICE_BASE_PATTERN
		$TOOLBOX slice 0 3 $NUM_SLICES z $BASE_DIR/$SCAN_FILE $WORK_DIR/$SLICE_PATTERN
	done
fi

if [[ $1 == *"2"* ]] ; then
	for SLICE_NUM in `seq 0 $(($NUM_SLICES-1))` ; do
		SLICE_FILES=$(find $WORK_DIR -name "*_$SLICE_BASE_PATTERN$SLICE_NUM*.ply")
		SLICE_NAME=$SLICE_BASE_PATTERN_$SLICE_NUM
		$TOOLBOX merge-cloud $SLICE_FILES $WORK_DIR/$SLICE_NAME
	done
fi

if [[ $1 == *"3"* ]] ; then
for SLICE_FILE in $(find $WORK_DIR -name "$SLICE_BASE_PATTERN*.ply") ; do
	CLUSTER_PATTERN=${SLICE_FILE%.ply}"_"$CLUSTER_BASE_PATTERN
	$TOOLBOX euclidian-clustering 0.03 60 10000 $SLICE_FILE $CLUSTER_PATTERN
done
fi

if [[ $1 == *"4"* ]] ; then
	for CLUSTER_FILE in $(find $WORK_DIR -name "*_$CLUSTER_BASE_PATTERN*" -printf "%f ") ; do
		BBOX_FILE=${CLUSTER_FILE/$CLUSTER_BASE_PATTERN/$BBOX_BASE_PATTERN}
		echo -e $BBOX_FILE
		$TOOLBOX get-bounding-box $WORK_DIR/$CLUSTER_FILE $WORK_DIR/$BBOX_FILE
	done
fi

if [[ "$1" == *"5"* ]] ; then
	BBOX_FILES=$(find $WORK_DIR -name "*_$BBOX_BASE_PATTERN*.ply")
	$TOOLBOX merge-mesh $BBOX_FILES $WORK_DIR/result.ply
fi

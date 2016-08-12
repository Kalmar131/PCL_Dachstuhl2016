#!/bin/bash

#./toolbox radius-outlier-removal 0.8 8 ~/Desktop/Studium/Belege/Objekterkennung/balken001/balken001.ply out.ply
#./toolbox statistical-outlier-removal 20 0.9 ~/Desktop/Studium/Belege/Objekterkennung/cluster_0.ply out.ply
#./toolbox moving-least-squares 0.02 ~/Desktop/Studium/Belege/Objekterkennung/balken001/balken001.ply out.ply

TOOLBOX=./toolbox/toolbox
WORK_DIR=../data/workflow1/
DATA_FILE=../data/balken001/balken001.ply
SLICE_PATTERN="slice_"
CLUSTER_BASE_PATTERN="cluster_"
PLANE_BASE_PATTERN="plane_"
LINE_BASE_PATTERN="line_"
MINMAX_BASE_PATTERN="minmax_"

if [[ $1 == *"1"* ]] ; then
	$TOOLBOX slice -1 3 10 z $DATA_FILE $WORK_DIR/$SLICE_PATTERN
fi

if [[ $1 == *"2"* ]] ; then
	for SLICE_FILE in $(find $WORK_DIR -name "$SLICE_PATTERN*.ply" -printf "%f ") ; do
		CLUSTER_PATTERN=${SLICE_FILE%.ply}"_"$CLUSTER_BASE_PATTERN
		$TOOLBOX euclidian-clustering 0.02 50 100000 $WORK_DIR/$SLICE_FILE $WORK_DIR/$CLUSTER_PATTERN
	done
fi

if [[ $1 == *"3"* ]] ; then
	echo "hello"
	for CLUSTER_FILE in $(find $WORK_DIR -name "*_$CLUSTER_BASE_PATTERN*.ply" -printf "%f ") ; do
		CLUSTER_PLANE_PATTERN=${CLUSTER_FILE%.ply}"_"$PLANE_BASE_PATTERN
		$TOOLBOX model-segment plane 0.01 100 $WORK_DIR/$CLUSTER_FILE $WORK_DIR/$CLUSTER_PLANE_PATTERN
	done
fi

if [[ $1 == *"4"* ]] ; then
	for PLANE_FILE in $(find $WORK_DIR -name "*_$PLANE_BASE_PATTERN*.ply" -printf "%f ") ; do
		HULL_FILE=${PLANE_FILE%.ply}"_hull.ply"
		$TOOLBOX concave-hull 0.1 $WORK_DIR/$PLANE_FILE $WORK_DIR/$HULL_FILE
	done
fi

if [[ $1 == *"5"* ]] ; then
	for HULL_FILE in $(find $WORK_DIR -name "*_hull.ply" -printf "%f ") ; do
		LINE_PATTERN=${HULL_FILE%_hull.ply}"_"$LINE_BASE_PATTERN
		$TOOLBOX model-segment line 0.01 10 $WORK_DIR/$HULL_FILE $WORK_DIR/$LINE_PATTERN
	done
fi

if [[ $1 == *"6"* ]] ; then
	for LINE_FILE in $(find $WORK_DIR -name "*_$LINE_BASE_PATTERN*.ply" -printf "%f ") ; do
		MINMAX_FILE=${LINE_FILE/$LINE_BASE_PATTERN/$MINMAX_BASE_PATTERN}
		$TOOLBOX min-max $WORK_DIR/$LINE_FILE $WORK_DIR/$MINMAX_FILE
	done
fi

if [[ $1 == *"7"* ]] ; then
	MINMAX_FILES=$(find $WORK_DIR -name "*_$MINMAX_BASE_PATTERN*.ply")
	$TOOLBOX merge-cloud $MINMAX_FILES $WORK_DIR/minmax.ply
fi

if [[ $1 == *"8"* ]] ; then
	$TOOLBOX create-edges $WORK_DIR/minmax.ply $WORK_DIR/result.ply
fi

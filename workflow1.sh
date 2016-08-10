#!/bin/bash

#./toolbox radius-outlier-removal 0.8 8 ~/Desktop/Studium/Belege/Objekterkennung/balken001/balken001.ply out.ply
#./toolbox statistical-outlier-removal 20 0.9 ~/Desktop/Studium/Belege/Objekterkennung/cluster_0.ply out.ply
#./toolbox moving-least-squares 0.02 ~/Desktop/Studium/Belege/Objekterkennung/balken001/balken001.ply out.ply

TOOLBOX=./toolbox/toolbox
BASE_DIR=../data/balken001/
DATA_FILE=balken001.ply
SLICE_PATTERN="slice_"
CLUSTER_BASE_PATTERN="cluster_"
PLANE_BASE_PATTERN="plane_"
LINE_BASE_PATTERN="line_"

$TOOLBOX slice 10 x $BASE_DIR/$DATA_FILE $BASE_DIR/$SLICE_PATTERN

for SLICE_FILE in $(find $BASE_DIR -name "$SLICE_PATTERN*" -printf "%f ") ; do
	echo $SLICE_FILE
	CLUSTER_PATTERN=${SLICE_FILE%.ply}"_"$CLUSTER_BASE_PATTERN
	$TOOLBOX euclidian-clustering 0.02 30 100000 $BASE_DIR/$SLICE_FILE $BASE_DIR/$CLUSTER_PATTERN

	for CLUSTER_FILE in $(find $BASE_DIR -name "$CLUSTER_PATTERN*" -printf "%f ") ; do
		CLUSTER_PLANE_PATTERN=${CLUSTER_FILE%.ply}"_"$PLANE_BASE_PATTERN
		$TOOLBOX model-segment plane 0.01 100 $BASE_DIR/$CLUSTER_FILE $BASE_DIR/$CLUSTER_PLANE_PATTERN

		for PLANE_FILE in $(find $BASE_DIR -name "$CLUSTER_PLANE_PATTERN*" -printf "%f ") ; do
			$TOOLBOX concave-hull 0.1 $BASE_DIR/$PLANE_FILE $BASE_DIR/${PLANE_FILE%.ply}"_hull.ply"
		done

		for HULL_FILE in $(find $BASE_DIR -name "*_hull.ply" -printf "%f ") ; do
			echo $HULL_FILE
			LINE_PATTERN=${HULL_FILE%_hull.ply}"_"$LINE_BASE_PATTERN
			$TOOLBOX model-segment line 0.01 10 $BASE_DIR/$HULL_FILE $BASE_DIR/$LINE_PATTERN
		done

	done

done

LINE_FILES=$(find $BASE_DIR -name "*$LINE_BASE_PATTERN*.ply")
$TOOLBOX create-mesh $LINE_FILES $BASE_DIR/lines.ply

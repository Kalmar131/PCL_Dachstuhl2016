#!/bin/bash

TOOLBOX=./toolbox/toolbox
NUM_SLICES=32

exclude_list=( 1 4 22 25 33 38 41 50 )

contains()
{
	j=0
	while (( j < ${#exclude_list[*]} )) ; do
  	  if (( $1 == ${exclude_list[$j]} )); then
    	  return 1
    	fi
		let j++
	done

	return 0
}

for i in `seq 1 55` ; do
	mkdir ../data/balken$i
	mkdir ../data/workflow5-balken$i
	cp ./Extracting/build/balken_orginal$i.ply ../data/balken$i/balken$i.ply
	$TOOLBOX statistical-outlier-removal 20 0.9 ../data/balken18/balken18.ply out.ply

	WORK_DIR=../data/workflow5-balken$i/
	DATA_FILE=../data/balken$i/balken$i.ply

	rm -rf $WORK_DIR/*
	$TOOLBOX reconstruct-bars 123456 z -1 3 $NUM_SLICES  0.1 60 10000 0.001 2000 0.2 0.04 $DATA_FILE $WORK_DIR/result $WORK_DIR/bar_ 
done

cp ../data/empty.ply ../data/result.ply	
for n in `seq 1 55` ; do
	contains $n
	if (( $? == 0 )) ; then
		$TOOLBOX merge-cloud withEdges ../data/workflow5-balken$n/result6.ply ../data/result.ply ../data/result.ply
	fi
done

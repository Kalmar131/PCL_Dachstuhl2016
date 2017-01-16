#!/bin/bash

TOOLBOX=./toolbox/toolbox
NUM_SLICES=32

reconstruct()
{
	WORK_DIR=../data/workflow5-$1/
	DATA_FILE=../data/$1/$1.ply

	rm -rf $WORK_DIR/*
	$TOOLBOX reconstruct-bars $2 z -1 3 $NUM_SLICES  0.1 60 10000 0.001 2000 0.2 0.04 $DATA_FILE $WORK_DIR/result $WORK_DIR/bar_ 
}

for i in `seq 1 55` ; do
	mkdir ../data/balken$i
	mkdir ../data/workflow5-balken$i
	cp ./Extracting/build/balken_orginal$i.ply ../data/balken$i/balken$i.ply
	reconstruct balken$i 123456
done

exclude_list=( 1 4 22 25 33 38 41 50 )

contains()
{
	i=0
	while (( i < ${#exclude_list[*]} )) ; do
  	  if (( $1 == ${exclude_list[$i]} )); then
    	  return 1
    	fi
		let i++
	done

	return 0
}

cp ../data/empty.ply ../data/result.ply	
for n in `seq 1 55` ; do
	contains $n
	if (( $? == 0 )) ; then
		$TOOLBOX merge-cloud withEdges ../data/workflow5-balken$n/result6.ply ../data/result.ply ../data/result.ply
	fi
done

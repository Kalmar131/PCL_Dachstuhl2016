#!/bin/bash

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
		./toolbox/toolbox merge-cloud withEdges ../data/workflow5-balken$n/result6.ply ../data/result.ply ../data/result.ply
	fi
done

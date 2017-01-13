#!/bin/bash

for i in `seq 1 55` ; do
	mkdir ../data/balken$i
	mkdir ../data/workflow5-balken$i
	cp ./Extracting/build/balken_orginal$i.ply ../data/balken$i/balken$i.ply
	./workflow5.sh balken$i 123456
done


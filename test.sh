./toolbox/toolbox radius-outlier-removal 0.8 8 ../data/workflow1/balken001.ply out.ply
./toolbox/toolbox statistical-outlier-removal 20 0.9 ../data/workflow1/balken001.ply out.ply
./toolbox/toolbox moving-least-squares 0.02 ../data/workflow1/balken001.ply out.ply
./toolbox/toolbox extract-bars z -1 3 32 0.02 50 100000 0.1 ../data/workflow1/balken001.ply
./toolbox/toolbox slice -0.5 0.5 1 z data/balken001/balken001.ply out.ply
./toolbox/toolbox euclidian-clustering 0.2 20 100000 slice_0.ply cluster_
./toolbox/toolbox model-segment plane 0.005 100 ../../data/test1/cluster_2_msl.ply ../../data/test1/cluster_2_plane_
./toolbox/toolbox merge-cloud ../../data/test1/cluster_2_plane_0.ply ../../data/test1/cluster_2_plane_1.ply ../../data/test1/cluster_2_planes.ply
./toolbox/toolbox get-bounding-box ../../data/test1/cluster_2_planes.ply ../../data/test1/bbox_2.ply
./toolbox/toolbox extract-sides 1000 0.1 0.02 400 50 0.95 0.01 ../../data/workflow1/slice_7_cluster_0.ply $WORK_DIR/result2.ply

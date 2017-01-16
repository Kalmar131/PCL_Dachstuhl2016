./toolbox slice -0.5 0.5 1 z data/balken001/balken001.ply out.ply
./toolbox euclidian-clustering 0.2 20 100000 slice_0.ply cluster_
./toolbox model-segment plane 0.005 100 ../../data/test1/cluster_2_msl.ply ../../data/test1/cluster_2_plane_
./toolbox merge-cloud ../../data/test1/cluster_2_plane_0.ply ../../data/test1/cluster_2_plane_1.ply ../../data/test1/cluster_2_planes.ply
./toolbox get-bounding-box ../../data/test1/cluster_2_planes.ply ../../data/test1/bbox_2.ply



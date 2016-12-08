#include <iostream>

#include <vector>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/organized.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/time.h>
#include <pcl/common/transforms.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/features/don.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;
using namespace pcl;

int filter(std::string file)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_s (new pcl::PointCloud<pcl::PointXYZ>);

	// Fill in the cloud data
	pcl::io::loadPLYFile(file, *cloud);

	// Create the filtering object
	// schneide nur die Balkenstümpfe heraus, um diese zu filtern
	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud (cloud);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits (0.1, 3.12);
	pass.filter (*cloud_filtered);

	//Stümpfe für DON
	pcl::PassThrough<pcl::PointXYZ> pass_f;
	pass_f.setInputCloud (cloud_filtered);
	pass_f.setFilterFieldName ("z");
	pass_f.setFilterLimits (1, 1.5);
	pass_f.filter (*cloud_s);

	pcl::io::savePLYFile("balken_stuempfe.ply", *cloud_s, false);

	std::cerr << "Cloud after filtering: " << std::endl;
	for (size_t i = 0; i < cloud_filtered->points.size (); ++i)
	std::cerr << "    " << cloud_filtered->points[i].x << " "
					<< cloud_filtered->points[i].y << " "
					<< cloud_filtered->points[i].z << std::endl;

	//Ausßreiser eleminieren
	// Create the filtering object
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud (cloud_filtered);
	sor.setMeanK (500);
	sor.setStddevMulThresh (1.0);
	sor.filter (*cloud_filtered);

	std::cerr << "Cloud after filtering: " << std::endl;
	std::cerr << *cloud_filtered << std::endl;

	pcl::io::savePLYFile("balkenwerk.ply", *cloud_filtered, false);

	  sor.setNegative (true);
	  sor.filter (*cloud_filtered);
	  pcl::io::savePLYFile("balkenwerk_outliers.ply", *cloud_filtered, false);

	return (0);

}

float distancexy(pcl::PointXYZ p1, pcl::PointXYZ p2)
{
	return (p1.x - p2.x)* (p1.x - p2.x) +  (p1.y - p2.y)* (p1.y - p2.y) ;
}

int euclidean_cluster_extraction (std::string file){

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);

	pcl::io::loadPLYFile(file,*cloud);

	std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*

	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (cloud);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance (0.15);
	ec.setMinClusterSize (100);
	ec.setMaxClusterSize (2000);
	ec.setSearchMethod (tree);
	ec.setInputCloud (cloud);
	ec.extract (cluster_indices);

	std::vector<pcl::PointXYZ> list;
	int j = 0;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
	{
		Eigen::Vector4f centroid;

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
			cloud_cluster->points.push_back (cloud->points[*pit]); //*

		cloud_cluster->width = cloud_cluster->points.size ();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		pcl::compute3DCentroid(*cloud_cluster,centroid);

		pcl::PointXYZ central_point;
		central_point.x = centroid[0];
		central_point.y = centroid[1];
		central_point.z = centroid[2];
		list.push_back(central_point);


		std::stringstream ss;
		ss << "cloud_cluster_" << j << ".ply";
		pcl::io::savePLYFile(ss.str (), *cloud_cluster, false); //*

	   j++;
	}

	pcl::PointCloud<PointXYZ>::Ptr cloudAll;
	pcl::PCDReader reader;
	reader.read("gesamt.pcd",*cloudAll);
	printf("size:%d\n", cloudAll->points.size());

	std::vector<pcl::PointIndices> indices(list.size());
	for (unsigned i = 0; i < cloudAll->points.size(); ++i)
	{
		float minDist = 1000;
		unsigned minIndex = 0;

		printf("i:%d", i);

		for (unsigned c = 0; c < list.size(); ++c)
		{
			pcl::PointXYZ p1 = list[c];
			pcl::PointXYZ p2 = cloudAll->points[i];
			float dist = distancexy(p1, p2);

			if (dist < minDist)
			{
				minDist = dist;
				minIndex = c;
			}
		}

		indices[minIndex].indices.push_back(i);
	}

	for (unsigned i = 0; i < indices.size(); ++i)
	{
		pcl::PointIndices& index = indices[i];

	      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator it = index.indices.begin (); it != index.indices.end (); ++it)
	    {
	        cloud_cluster->points.push_back(cloud->points[*it]);
	    }

		pcl::PLYWriter writer;
	    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
	    std::stringstream ss;
	    ss << "cloud_cluster_" << i << ".ply";
	    writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false); //*
	}


	return (0);
}

int mint_cut(std::string file){

	  pcl::PointCloud <pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud <pcl::PointXYZ>);
	  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_mint (new pcl::PointCloud<pcl::PointXYZ>);
	  if ( pcl::io::loadPLYFile(file, *cloud) == -1 )
	  {
	    std::cout << "Cloud reading failed." << std::endl;
	    return (-1);
	  }
	  pcl::MinCutSegmentation<pcl::PointXYZ> seg;
	  seg.setInputCloud (cloud);

	  pcl::PointCloud<pcl::PointXYZ>::Ptr foreground_points(new pcl::PointCloud<pcl::PointXYZ> ());
	  pcl::PointXYZ point;
	  point.x = 3;
	  point.y =3;
	  point.z = 1;
	  foreground_points->points.push_back(point);
	  seg.setForegroundPoints (foreground_points);

	  seg.setSigma (0.25);
	  seg.setRadius (1.5);
	  seg.setNumberOfNeighbours (5);
	  seg.setSourceWeight (0.8);

	  std::vector <pcl::PointIndices> clusters;
	  seg.extract (clusters);

	  std::cout << "Maximum flow is " << seg.getMaxFlow () << std::endl;

	  //Extraxting the cluster
	  	  int j = 0;
	  	    for (std::vector<pcl::PointIndices>::const_iterator it = clusters.begin (); it != clusters.end (); ++it)
	  	    {
	  	      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
	  	      for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
	  	        cloud_cluster->points.push_back (cloud->points[*pit]); //*
	  	      cloud_cluster->width = cloud_cluster->points.size ();
	  	      cloud_cluster->height = 1;
	  	      cloud_cluster->is_dense = true;

	  	      std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
	  	      if(j==1){
	  	    	  pcl::io::savePLYFile("mint_cut.ply", *cloud_cluster, false); //*
	  	      }
	  	      j++;
	  	    }

	return(0);
}

int main()
{
	//filter("ScanPos08_Scan01.ply");
	//don_extracting("balken_stuempfe.ply");
	euclidean_cluster_extraction("balken_stuempfe_cut.ply");
	//mint_cut("balkenwerk.ply");
	cout <<" ready";

	return (0);
}

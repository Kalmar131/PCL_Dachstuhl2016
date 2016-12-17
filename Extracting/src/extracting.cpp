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

	pcl::PCDReader reader;
	// Fill in the cloud data
	reader.read(file, *cloud);

	// Create the filtering object
	// schneide nur die Balkenstümpfe heraus, um diese zu filtern
	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud (cloud);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits (0.1, 3.12);
	pass.filter (*cloud_filtered);

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

int euclidean_cluster_extraction (std::string file){

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_stuempfe (new pcl::PointCloud<pcl::PointXYZ>);

	pcl::io::loadPLYFile(file,*cloud_stuempfe);

	std::cout << "PointCloud before filtering has: " << cloud_stuempfe->points.size () << " data points." << std::endl; //*

	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (cloud_stuempfe);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance (0.2);
	ec.setMinClusterSize (40);
	ec.setMaxClusterSize (200);
	ec.setSearchMethod (tree);
	ec.setInputCloud (cloud_stuempfe);
	ec.extract (cluster_indices);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_centroid (new pcl::PointCloud<pcl::PointXYZ>);
	int j = 0;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
	{
		Eigen::Vector4f centroid;
		//Globale Variablen

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
			cloud_cluster->points.push_back (cloud_stuempfe->points[*pit]); //*

		cloud_cluster->width = cloud_cluster->points.size ();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		pcl::compute3DCentroid(*cloud_cluster,centroid);

		pcl::PointXYZ central_point;
		central_point.x = centroid[0];
		central_point.y = centroid[1];
		central_point.z = centroid[2];
		cloud_centroid->points.push_back(central_point);

		std::stringstream ss;
		ss << "cloud_cluster_" << j << ".ply";
		pcl::io::savePLYFile(ss.str (), *cloud_cluster, false); //*

	   j++;
	}
	pcl::io::savePLYFile("center.ply", *cloud_centroid, false); //Für Test ob Cenbtroide auch wirklich die Mittelpunkte sind

	return (0);
}

int min_cut(std::string gesamt,std::string centroide){

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_centroid (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_center_filtered (new pcl::PointCloud<pcl::PointXYZ>);

	pcl::io::loadPLYFile(gesamt,*cloud_in);
	pcl::io::loadPLYFile(centroide,*cloud_centroid);

	pcl::PassThrough<pcl::PointXYZ> pass_center;
	pass_center.setInputCloud (cloud_centroid);
	pass_center.setFilterFieldName ("y");
	pass_center.setFilterLimits (1, 13.5);
	pass_center.filter (*cloud_center_filtered);

	pcl::io::savePLYFile("cloud_center_filtered.ply", *cloud_center_filtered, false);

	int x=0;
	for (pcl::PointCloud<pcl::PointXYZ>::const_iterator mx = cloud_center_filtered->points.begin();mx != cloud_center_filtered->points.end(); ++mx)
	{
	// Min-cut clustering object.
		pcl::MinCutSegmentation<pcl::PointXYZ> clustering;
		clustering.setInputCloud(cloud_in);
		// Create a cloud that lists all the points that we know belong to the object
		// (foreground points). We should set here the object's center.
		pcl::PointCloud<pcl::PointXYZ>::Ptr foregroundPoints(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointXYZ point;
		point.x=mx->x;
		point.y=mx->y;
		point.z=0;
		std::cout << "CUT X: " << point.x << std::endl;
		std::cout << "CUT Y: " << point.y << std::endl;
		foregroundPoints->points.push_back(point);
		clustering.setForegroundPoints(foregroundPoints);
		// Set sigma, which affects the smooth cost calculation. It should be
		// set depending on the spacing between points in the cloud (resolution).
		clustering.setSigma(0.05);
		// Set the radius of the object we are looking for.
		clustering.setRadius(1.5);
		// Set the number of neighbors to look for. Increasing this also increases
		// the number of edges the graph will have.
		clustering.setNumberOfNeighbours(20);
		// Set the foreground penalty. It is the weight of the edges
		// that connect clouds points with the source vertex.
		clustering.setSourceWeight(0.8);

		std::vector <pcl::PointIndices> clusters;
		clustering.extract(clusters);
		// For every cluster...

		for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
		{
			// ...add all its points to a new cloud...
			pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_mint(new pcl::PointCloud<pcl::PointXYZ>);
			for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
				cluster_mint->points.push_back(cloud_in->points[*point]);
			cluster_mint->width = cluster_mint->points.size();
			cluster_mint->height = 1;
			cluster_mint->is_dense = true;

				std::stringstream sm;
				sm << "min_cut_" << x << ".ply";
				pcl::io::savePLYFile(sm.str (), *cluster_mint, false);
				std::cout << sm.str() << "extracted"<< std::endl;
				x++;

		}
	}

	return(0);
}

int main()
{
	filter("gesamt.pcd");
	euclidean_cluster_extraction("balken_stuempfe.ply");
	min_cut("balkenwerk.ply","center.ply");
	cout <<" ready";

	return (0);
}

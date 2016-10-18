#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <string>
#include <vector>

using namespace std;

class Filter
{
	public:
		int downsampling(string file);
		int groundfilter(string file);
		//int outlier_removal(string file); TODO: Noch nicht fertig
};

int Filter::downsampling(string file)
{
	pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2 ());
	pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2 ());

	pcl::PCDReader reader;
	// Einlsesen der Punktwolke
	reader.read (file, *cloud);

	std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height
	   << " data points (" << pcl::getFieldsList (*cloud) << ").";

	// der Filter wird erstellt (Voxel)
	pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
	sor.setInputCloud (cloud);
	sor.setLeafSize (0.1f, 0.1f, 0.1f);
	sor.filter (*cloud_filtered);

	std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height
	   << " data points (" << pcl::getFieldsList (*cloud_filtered) << ").";

	pcl::PCDWriter writer;
	writer.write ("downsampled_"+file, *cloud_filtered,
		 Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), false);

	return (0);
}

int Filter::groundfilter(string file){

	file = "downsampled_"+file;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointIndicesPtr ground (new pcl::PointIndices);

	// Fill in the cloud data
	pcl::PCDReader reader;
	// Replace the path below with the path where you saved your file
	reader.read<pcl::PointXYZ> (file, *cloud);

	std::cerr << "Cloud before filtering: " << std::endl;
	std::cerr << *cloud << std::endl;

	// Create the filtering object
	pcl::ProgressiveMorphologicalFilter<pcl::PointXYZ> pmf;
	pmf.setInputCloud (cloud);
	pmf.setMaxWindowSize (20);
	pmf.setSlope (0.1f);
	pmf.setInitialDistance (0.5f);
	pmf.setMaxDistance (3.0f);
	pmf.extract (ground->indices);

	// Create the filtering object
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	extract.setInputCloud (cloud);
	extract.setIndices (ground);
	extract.filter (*cloud_filtered);

	std::cerr << "Ground cloud after filtering: " << std::endl;
	std::cerr << *cloud_filtered << std::endl;

	pcl::PCDWriter writer;
	writer.write<pcl::PointXYZ> ("ground_"+file, *cloud_filtered, false);

	// Extract non-ground returns
	extract.setNegative (true);
	extract.filter (*cloud_filtered);

	std::cerr << "Object cloud after filtering: " << std::endl;
	std::cerr << *cloud_filtered << std::endl;

	writer.write<pcl::PointXYZ> ("utm_object_"+file, *cloud_filtered, false);

	return (0);
}

int main(int argc, char** argv)
{
	Filter filter;
	std::vector<std::string> input;
	input.push_back("scan8.pcd");
	input.push_back("scan12.pcd");

	for(vector<string>::const_iterator i = input.begin(); i != input.end(); ++i) {

		filter.downsampling(*i);
		filter.groundfilter(*i);
	}

cout <<"ready";
}




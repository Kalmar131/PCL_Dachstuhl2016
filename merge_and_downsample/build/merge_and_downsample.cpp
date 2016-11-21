// Merges multiple point clouds and downsamples the result

#include <iostream>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

// downsample cloud data using VoxelGrid filter
// based on http://pointclouds.org/documentation/tutorials/voxel_grid.php
void downsample(const std::string& infile, const std::string& outfile, float leafsize) {
  std::cout << "Downsampling " << infile << " ..." << std::endl;

  // fill in the cloud data
  pcl::PCLPointCloud2::Ptr cloud_blob(new pcl::PCLPointCloud2), cloud_filtered_blob(new pcl::PCLPointCloud2);
  pcl::PCDReader reader;
  reader.read(infile, *cloud_blob);

  int points_before = cloud_blob->width * cloud_blob->height;
  std::cout << "PointCloud before filtering: " << points_before << " data points." << std::endl;

  // create the filtering object: downsample the dataset using the given leaf size
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud(cloud_blob);
  sor.setLeafSize(leafsize, leafsize, leafsize);
  sor.filter(*cloud_filtered_blob);

  // convert to the templated PointCloud
  pcl::fromPCLPointCloud2(*cloud_filtered_blob, *cloud_filtered);

  int points_after = cloud_filtered->width * cloud_filtered->height;
  std::cout << "PointCloud after filtering: " << points_after << " data points (" << (points_after / (float)points_before) * 100.0f << "%)." << std::endl;

  // write the downsampled version to disk
  pcl::PCDWriter writer;
  writer.write<pcl::PointXYZ>(outfile, *cloud_filtered, false);
}

// merges a list of point cloud files
pcl::PointCloud<pcl::PointXYZ>::Ptr merge(const std::vector<std::string>& infiles, const std::string& outfile) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_merged(new pcl::PointCloud<pcl::PointXYZ>);

  for(std::vector<std::string>::const_iterator infile = infiles.begin(); infile != infiles.end(); ++infile) {
    // read input file
    std::cout << "Reading PointCloud " << *infile << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (-1 == pcl::io::loadPCDFile<pcl::PointXYZ>(*infile, *cloud)) {
      std::cerr << "Couldn't read point cloud" << std::endl;
      exit(1);
    }
    std::cout << "PointCloud " << *infile << ": width: " << cloud->width << " height: " << cloud->height << " points: " << cloud->width * cloud->height << std::endl;

    // add (merge)
    *cloud_merged += *cloud;
  }

  // write merge result
  std::cout << "Merged PointCloud: width: " << cloud_merged->width << " height: " << cloud_merged->height << " points: " << cloud_merged->width * cloud_merged->height << std::endl;
  pcl::PCDWriter writer;
  writer.write<pcl::PointXYZ>(outfile, *cloud_merged, true /* binary? */);
}


int main(int argc, char** argv) {
  if (argc <= 1) {
    std::cerr << "Please specify a list of files to merge" << std::endl;
    exit(1);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_merged(new pcl::PointCloud<pcl::PointXYZ>);
  for (int i = 1; i < argc; ++i) {
    // read input file
    std::cout << "Reading PointCloud " << argv[i] << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (-1 == pcl::io::loadPCDFile<pcl::PointXYZ>(argv[i], *cloud)) {
      std::cerr << "Couldn't read point cloud" << std::endl;
      exit(1);
    }
    std::cout << "PointCloud " << argv[i] << ": width: " << cloud->width << " height: " << cloud->height << " points: " << cloud->width * cloud->height << std::endl;

    // add (merge)
    *cloud_merged += *cloud;
  }

  // write merge result
  std::cout << "Merged PointCloud: width: " << cloud_merged->width << " height: " << cloud_merged->height << " points: " << cloud_merged->width * cloud_merged->height << std::endl;
  pcl::PCDWriter writer;
  writer.write<pcl::PointXYZ>("merged.pcd", *cloud_merged, true /* binary? */);

  // downsample merge result
  std::cout << "Downsampling for leaf size 0.05" << std::endl;
  downsample("merged.pcd", "merged_downsampled_0.05.pcd", 0.05f);
  std::cout << "Downsampling for leaf size 0.1" << std::endl;
  downsample("merged.pcd", "merged_downsampled_0.1.pcd", 0.1f);
  std::cout << "Downsampling for leaf size 0.2" << std::endl;
  downsample("merged.pcd", "merged_downsampled_0.2.pcd", 0.2f);
}


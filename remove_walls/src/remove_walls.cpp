// tries to remove ground and all walls from the building

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/progressive_morphological_filter.h>

// reads cloud file from disk and returns it
pcl::PointCloud<pcl::PointXYZ>::Ptr load_cloud(const std::string& infile) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PCDReader reader;
  if (-1 == reader.read<pcl::PointXYZ>(infile, *cloud)) {
    std::cerr << "Couldn't read point cloud " << infile << std::endl;
    exit(1);
  }
  std::cout << "PointCloud " << infile << " loaded: width: " << cloud->width << " height: " << cloud->height << " points: " << cloud->width * cloud->height << std::endl;
  return cloud;
}

// writes cloud to disk
void write_cloud(const std::string& outfile, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
  pcl::PCDWriter writer;
  writer.write<pcl::PointXYZ>(outfile, *cloud, false /* binary */);
  std::cout << "PointCloud " << outfile << " written" << std::endl;
}

// rotates a point cloud based on the given rotation angle (in degree) and rotation axis
// based on http://pointclouds.org/documentation/tutorials/matrix_transform.php
void rotate_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float angle, const Eigen::Vector3f& axis) {
  // convert angle from degree to radians
  float angle_radians = angle * M_PI / 180.0;

  // set transformation parameters
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  transform.rotate(Eigen::AngleAxisf(angle_radians, axis));

  // execute the transformation
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::transformPointCloud(*cloud, *transformed_cloud, transform);
  *cloud = *transformed_cloud;
}

// removes the ground using ProgressiveMorphologicalFilter from the given cloud
// based on http://pointclouds.org/documentation/tutorials/progressive_morphological_filtering.php
void remove_ground(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr ground) {
  // set ground filter parameters
  pcl::PointIndicesPtr ground_indices(new pcl::PointIndices);
  pcl::ProgressiveMorphologicalFilter<pcl::PointXYZ> pmf;
  pmf.setInputCloud(cloud);
  pmf.setMaxWindowSize(20);
  pmf.setSlope(0.1f);
  pmf.setInitialDistance(0.5f);
  pmf.setMaxDistance(3.0f);

  // compute ground indices
  pmf.extract(ground_indices->indices);

  // store ground in separate cloud
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(ground_indices);
  extract.filter(*ground);

  // remove ground from original cloud
  extract.setNegative(true);
  extract.filter(*cloud);
}

// little helper function for performing the steps rotate, extract and rotate back
void rotate_and_extract_ground(int step, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float angle, Eigen::Vector3f axis) {
  // file name prefix for storing of intermediate results
  std::stringstream num;
  num << step;
  std::string outfile_prefix = "dbg_step" + num.str();

  // rotate cloud, if required
  if (angle != 0.0f) {
    rotate_cloud(cloud, angle, axis);

    // write rotation result for debugging
    write_cloud(outfile_prefix + "_cloud_rotated.pcd", cloud);
  }

  // remove ground
  pcl::PointCloud<pcl::PointXYZ>::Ptr ground(new pcl::PointCloud<pcl::PointXYZ>);
  remove_ground(cloud, ground);

  // rotate back, if required
  if (angle != 0.0f) {
    rotate_cloud(cloud,  -angle, axis);
    rotate_cloud(ground, -angle, axis);
  }

  // write resulting cloud and extracted ground for debugging
  write_cloud(outfile_prefix + "_result.pcd", cloud);
  write_cloud(outfile_prefix + "_ground.pcd", ground);
}

int main(int argc, char** argv) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = load_cloud("merged_downsampled.pcd");

  int step = 1;

  // extract ground
  std::cout << "Extracting ground, step: " << step << std::endl;
  // (angle of 0.0 to skip rotation)
  rotate_and_extract_ground(step++, cloud, 0.0, Eigen::Vector3f::UnitX());

  // extract wall
  std::cout << "Extracting wall, step: " << step << std::endl;
  rotate_and_extract_ground(step++, cloud, 130, Eigen::Vector3f::UnitX());

  // extract other wall
  std::cout << "Extracting other wall, step: " << step << std::endl;
  rotate_and_extract_ground(step++, cloud, -130, Eigen::Vector3f::UnitX());

  // extract back wall
  std::cout << "Extracting back wall, step: " << step << std::endl;
  rotate_and_extract_ground(step++, cloud, 90, Eigen::Vector3f::UnitY());

  // extract back wall, again
  std::cout << "Extracting back wall, step: " << step << std::endl;
  rotate_and_extract_ground(step++, cloud, 90, Eigen::Vector3f::UnitY());

  // extract cylindrical wall
  std::cout << "Extracting cylindrical wall, step: " << step << std::endl;
  rotate_and_extract_ground(step++, cloud, -90, Eigen::Vector3f::UnitY());

  // extract cylindrical wall, again
  std::cout << "Extracting cylindrical wall, step: " << step << std::endl;
  rotate_and_extract_ground(step++, cloud, -90, Eigen::Vector3f::UnitY());

  write_cloud("result.pcd", cloud);
}


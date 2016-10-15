// Extracts ground, roof and wall planes

#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

// downsample cloud data using VoxelGrid filter
void downsample(const std::string& infile, const std::string& outfile, float leafsize) {
  std::cout << "Downsampling " << infile << " ..." << std::endl;

  // fill in the cloud data
  pcl::PCLPointCloud2::Ptr cloud_blob(new pcl::PCLPointCloud2), cloud_filtered_blob(new pcl::PCLPointCloud2);
  pcl::PCDReader reader;
  reader.read(infile, *cloud_blob);

  std::cout << "PointCloud before filtering: " << cloud_blob->width * cloud_blob->height << " data points." << std::endl;

  // create the filtering object: downsample the dataset using the given leaf size
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud(cloud_blob);
  sor.setLeafSize(leafsize, leafsize, leafsize);
  sor.filter(*cloud_filtered_blob);

  // convert to the templated PointCloud
  pcl::fromPCLPointCloud2(*cloud_filtered_blob, *cloud_filtered);

  std::cout << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height << " data points." << std::endl;

  // write the downsampled version to disk
  pcl::PCDWriter writer;
  writer.write<pcl::PointXYZ>(outfile, *cloud_filtered, false);
}

// segment planes using SACSegmentation
// note: eps_angle must be != 0.0 for axis to be actually used by the model
void segment_planes(const std::string& infile, const std::string& outfile_prefix, int max_planes, pcl::SacModel sac_model, double dist_threshold, double eps_angle, const Eigen::Vector3f& axis) {
  std::cout << "Segmenting planes ..." << std::endl;

  // fill in the cloud data
  pcl::PCLPointCloud2::Ptr cloud_blob(new pcl::PCLPointCloud2);
  pcl::PCDReader reader;
  reader.read(infile, *cloud_blob);

  // convert to the templated PointCloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>), cloud_p(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(*cloud_blob, *cloud_filtered);

  // create the segmentation object and set segmentation parameters
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(sac_model);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(1000);
  seg.setDistanceThreshold(dist_threshold);
  seg.setEpsAngle(eps_angle);
  seg.setAxis(axis);

  // create the filtering object
  pcl::ExtractIndices<pcl::PointXYZ> extract;

  int nr_points = (int) cloud_filtered->points.size();
  // extract the nth largest planar components from the remaining cloud
  for(int i = 0; i < max_planes; ++i) {
    // segment the largest planar component from the remaining cloud
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    seg.setInputCloud(cloud_filtered);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size() == 0) {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // extract the inliers
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*cloud_p);
    std::cout << "PointCloud " << i << " representing the planar component: " << cloud_p->width * cloud_p->height << " data points." << std::endl;
    std::cout << "Model coefficients: " << coefficients->values[0] << " " 
                                        << coefficients->values[1] << " "
                                        << coefficients->values[2] << " " 
                                        << coefficients->values[3] << std::endl;

    // write the plane to disk
    std::stringstream ss;
    ss << outfile_prefix << "_" << i << ".pcd";
    pcl::PCDWriter writer;
    writer.write<pcl::PointXYZ>(ss.str(), *cloud_p, false);

    // create the filtering object for the next iteration
    extract.setNegative(true);
    extract.filter(*cloud_f);
    cloud_filtered.swap(cloud_f);
  }
}

int main(int argc, char** argv) {
  // downsample input cloud to speed-up further processing
  downsample("scan8.pcd", "scan8_downsampled.pcd", 0.05f);

  // extract two horizontal planes (1st and 2nd floor)
  std::cout << std::endl;
  std::cout << "Extracting horizontal planes (1st and 2nd floor)" << std::endl;
  segment_planes("scan8_downsampled.pcd", "scan8_floor", 2, pcl::SACMODEL_PLANE, 0.05, 0.0, Eigen::Vector3f(0.0f, 0.0f, 1.0f));

  // extract rear wall (also extracts various other wrong planes :/)
  std::cout << std::endl;
  std::cout << "Extracting rear wall" << std::endl;
  segment_planes("scan8_downsampled.pcd", "scan8_rear_wall", 8, pcl::SACMODEL_PERPENDICULAR_PLANE, 0.1, 10.0 * (M_PI/180.0), Eigen::Vector3f(1.0f, -0.1f, 0.0f));

  // extract roofs (also extracts various other wrong planes :/)
  std::cout << std::endl;
  std::cout << "Extracting roof pitch 1" << std::endl;
  segment_planes("scan8_downsampled.pcd", "scan8_roof_pitch_1", 6, pcl::SACMODEL_PERPENDICULAR_PLANE, 0.05, 10.0 * (M_PI/180.0), Eigen::Vector3f(0.0f, 0.75f, 0.65f));
  std::cout << std::endl;
  std::cout << "Extracting roof pitch 2" << std::endl;
  segment_planes("scan8_downsampled.pcd", "scan8_roof_pitch_2", 3, pcl::SACMODEL_PERPENDICULAR_PLANE, 0.05, 10.0 * (M_PI/180.0), Eigen::Vector3f(0.0f, 0.63f, -0.76f));

  // accidentally extracts "rows of bars", maybe useful for something:
  //segment_planes("scan8_downsampled.pcd", "scan8_bar_row", 2, pcl::SACMODEL_PERPENDICULAR_PLANE, 0.05, 10.0 * (M_PI/180.0), Eigen::Vector3f(0.0f, 1.0f, 0.0f));
}


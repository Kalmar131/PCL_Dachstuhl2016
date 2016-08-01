#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/mls.h>

typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point> Cloud;

Cloud::Ptr passThrough(Cloud::Ptr cloudInput, const char* fieldName, float min, float max)
{
	Cloud::Ptr cloudOutput(new Cloud);

  pcl::PassThrough<Point> tool;
  tool.setInputCloud(cloudInput);
  tool.setFilterFieldName(fieldName);
  tool.setFilterLimits(min, max);
  tool.setFilterLimitsNegative(false);
  tool.filter(*cloudOutput);

	return cloudOutput;
}

Cloud::Ptr radiusOutlierRemoval(Cloud::Ptr cloudInput, float radiusSearch, int minNeighborsInRadius)
{
	Cloud::Ptr cloudOutput(new Cloud);

  pcl::RadiusOutlierRemoval<pcl::PointXYZ> tool;
  tool.setInputCloud(cloudInput);
  tool.setRadiusSearch(radiusSearch);
  tool.setMinNeighborsInRadius(minNeighborsInRadius);
  tool.filter(*cloudOutput);

	return cloudOutput;
}

Cloud::Ptr statisticalOutlierRemoval(Cloud::Ptr cloudInput, int meanK, float stddevMulThresh)
{
	Cloud::Ptr cloudOutput(new Cloud);

	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> tool;
  tool.setInputCloud(cloudInput);
  tool.setMeanK(meanK);
  tool.setStddevMulThresh(stddevMulThresh);
  tool.filter(*cloudOutput);

	return cloudOutput;
}


Cloud::Ptr concaveHull(Cloud::Ptr cloudInput, float alpha)
{
	Cloud::Ptr cloudOutput(new Cloud);

 	pcl::ConcaveHull<pcl::PointXYZ> tool;
 	tool.setInputCloud(cloudInput);
 	tool.setAlpha(0.1);
 	tool.reconstruct(*cloudOutput);

	return cloudOutput;
}

pcl::PointCloud<pcl::PointNormal>::Ptr movingLeastSquares(Cloud::Ptr cloudInput, float searchRadius)
{
  pcl::PointCloud<pcl::PointNormal>::Ptr cloudOutput(new pcl::PointCloud<pcl::PointNormal>);

  pcl::search::KdTree<Point>::Ptr tree (new pcl::search::KdTree<Point>);

  pcl::MovingLeastSquares<Point, pcl::PointNormal> tool;
  tool.setComputeNormals(true);
  tool.setInputCloud(cloudInput);
  tool.setPolynomialFit(true);
  tool.setSearchMethod(tree);
  tool.setSearchRadius(searchRadius);
  tool.process(*cloudOutput);

	return cloudOutput;
}

std::vector<Cloud::Ptr> euclidianClustering(Cloud::Ptr cloudInput, float clusterTolerance, int minClusterSize, int maxClusterSize)
{
	std::vector<Cloud::Ptr> listCloudOutput;

  pcl::search::KdTree<Point>::Ptr tree(new pcl::search::KdTree<Point>);
  tree->setInputCloud(cloudInput);

  std::vector<pcl::PointIndices> clusterIndices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> tool;
  tool.setClusterTolerance(clusterTolerance);
  tool.setMinClusterSize(minClusterSize);
  tool.setMaxClusterSize(maxClusterSize);
  tool.setSearchMethod(tree);
  tool.setInputCloud(cloudInput);
  tool.extract(clusterIndices);

  unsigned i = 0;
  for (std::vector<pcl::PointIndices>::const_iterator it = clusterIndices.begin (); it != clusterIndices.end (); ++it)
  {
    Cloud::Ptr cloud_cluster (new Cloud);

    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      cloud_cluster->points.push_back(cloudInput->points[*pit]);

    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

		listCloudOutput.push_back(cloud_cluster);
  }

  return listCloudOutput;
}

std::pair<pcl::PointIndices::Ptr, pcl::ModelCoefficients::Ptr> sacSegmentation(Cloud::Ptr cloudInput, int modelType, float threshold)
{
 	pcl::PointIndices::Ptr indices(new pcl::PointIndices);
 	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

 	pcl::SACSegmentation<pcl::PointXYZ> tool;
 	tool.setModelType(modelType);
 	tool.setMethodType(pcl::SAC_RANSAC);
 	tool.setDistanceThreshold(threshold);
 	tool.setInputCloud(cloudInput);
 	tool.setOptimizeCoefficients(true);
  tool.segment(*indices, *coefficients);
    
	return std::pair<pcl::PointIndices::Ptr, pcl::ModelCoefficients::Ptr>(indices, coefficients);
}

Cloud::Ptr projectInliers(Cloud::Ptr cloudInput, int modelType, pcl::ModelCoefficients::Ptr modelCoefficients)
{
  Cloud::Ptr cloudOutput(new Cloud);

	pcl::PointIndices::Ptr indices(new pcl::PointIndices);
	for (unsigned i = 0; i < cloudInput->points.size(); ++i)
		indices->indices.push_back(i);

 	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new Cloud);
	pcl::ProjectInliers<pcl::PointXYZ> tool;
	tool.setModelType(modelType);
	tool.setInputCloud(cloudInput);
	tool.setIndices(indices);
	tool.setModelCoefficients(modelCoefficients);
	tool.filter(*cloudOutput);

	return cloudOutput;
}

Cloud::Ptr filterByIndices(Cloud::Ptr cloudInput, pcl::PointIndices::Ptr indices, bool negative)
{
  Cloud::Ptr cloudOutput(new Cloud);

	pcl::ExtractIndices<pcl::PointXYZ> tool;
  tool.setInputCloud(cloudInput);
  tool.setIndices(indices);
  tool.setNegative(negative);
  tool.filter(*cloudOutput);

	return cloudOutput;
}

int getModelType(std::string modelName)
{
	int modelType;

	if (modelName == "plane")
		modelType = pcl::SACMODEL_PLANE;
	else if (modelName == "line")
		modelType = pcl::SACMODEL_LINE;

	return modelType;
}

pcl::ModelCoefficients::Ptr loadModelCoeffients(const char* filename)
{
	pcl::ModelCoefficients::Ptr modelCoefficients;
	return modelCoefficients;
}

void saveModelCoefficients(const char* modelFilename, pcl::ModelCoefficients::Ptr coefficents)
{
}

std::string createFilename(std::string filenameBase, unsigned seq)
{
	std::stringstream ss;
 	ss << filenameBase << seq << ".ply";

	return ss.str();
}

int main (int argc, char** argv)
{
	unsigned argn = 1;
	std::string toolname = argv[argn++];
	
	if (toolname == "pass-through")
	{
		const char* fieldName = argv[argn++];
		float min = atof(argv[argn++]);
		float max = atof(argv[argn++]);
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

  	pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

  	pcl::io::loadPLYFile(inFilename, *cloudInput);
  	std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		Cloud::Ptr cloudOutput = passThrough(cloudInput, fieldName, min, max);

	  std::cerr << "Output Cloud Size: " << cloudOutput->points.size() << std::endl;
  	pcl::io::savePLYFile(outFilename, *cloudOutput);
	}

	else if (toolname == "radius-outlier-removal")
	{
		float radiusSearch = atof(argv[argn++]);
		int minNeighborsInRadius = atoi(argv[argn++]);
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

  	pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

  	pcl::io::loadPLYFile(inFilename, *cloudInput);
  	std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		Cloud::Ptr cloudOutput = radiusOutlierRemoval(cloudInput, radiusSearch, minNeighborsInRadius);

	  std::cerr << "Output Cloud Size: " << cloudOutput->points.size() << std::endl;
  	pcl::io::savePLYFile(outFilename, *cloudOutput);
	}

	else if (toolname == "statistical-outlier-removal")
	{
		int meanK = atoi(argv[argn++]);
		float stddevMulThresh = atof(argv[argn++]);
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

  	pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

  	pcl::io::loadPLYFile(inFilename, *cloudInput);
  	std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		Cloud::Ptr cloudOutput = statisticalOutlierRemoval(cloudInput, meanK, stddevMulThresh);

	  std::cerr << "Output Cloud Size: " << cloudOutput->points.size() << std::endl;
  	pcl::io::savePLYFile(outFilename, *cloudOutput);
	}

	else if (toolname  == "concave-hull")
	{
		float alpha = atof(argv[argn++]);
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

  	pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

  	pcl::io::loadPLYFile(inFilename, *cloudInput);
  	std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		Cloud::Ptr cloudOutput = concaveHull(cloudInput, alpha);

	  std::cerr << "Output Cloud Size: " << cloudOutput->points.size() << std::endl;
  	pcl::io::savePLYFile(outFilename, *cloudOutput);
	}

	else if (toolname == "moving-least-squares")
	{
		float searchRadius = atof(argv[argn++]);
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

  	pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

  	pcl::io::loadPLYFile(inFilename, *cloudInput);
  	std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		pcl::PointCloud<pcl::PointNormal>::Ptr cloudOutput = movingLeastSquares(cloudInput, searchRadius);

	  std::cerr << "Output Cloud Size: " << cloudOutput->points.size() << std::endl;
  	pcl::io::savePLYFile(outFilename, *cloudOutput);
	}

	else if (toolname == "euclidian-clustering")
	{
		float clusterTolerance = atof(argv[argn++]);
		int minClusterSize = atoi(argv[argn++]);
		int maxClusterSize = atoi(argv[argn++]);
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

  	pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

  	pcl::io::loadPLYFile(inFilename, *cloudInput);
  	std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		std::vector<Cloud::Ptr> listCloudOutput = euclidianClustering(cloudInput, clusterTolerance, minClusterSize, maxClusterSize);

		for (unsigned i = 0; i < listCloudOutput.size(); ++i)
		{
  		std::cerr << "Index: " << i << " Output Cloud Size: " << listCloudOutput[i]->points.size() << std::endl;
  		pcl::io::savePLYFile(createFilename(outFilename, i), *(listCloudOutput[i]));
    }
	}
	else if (toolname == "sac-segmentation")
	{
		int modelType = getModelType(argv[argn++]);
		float threshold = atof(argv[argn++]);
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];
		const char* modelFilename = argv[argn++];

  	pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

  	pcl::io::loadPLYFile(inFilename, *cloudInput);
  	std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		std::pair<pcl::PointIndices::Ptr, pcl::ModelCoefficients::Ptr> pairIndicesAndCoefficents = sacSegmentation(cloudInput, modelType, threshold);

	  std::cerr << "Output Cloud Size: " << pairIndicesAndCoefficents.first->indices.size() << std::endl;
		saveModelCoefficients(modelFilename, pairIndicesAndCoefficents.second);
	}
	else if (toolname == "project-inliers")
	{
		int modelType = getModelType(argv[argn++]);
		pcl::ModelCoefficients::Ptr modelCoefficients = loadModelCoeffients(argv[argn++]);
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

  	pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

  	pcl::io::loadPLYFile(inFilename, *cloudInput);
  	std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		Cloud::Ptr cloudOutput = projectInliers(cloudInput, modelType, modelCoefficients);

	  std::cerr << "Output Cloud Size: " << cloudOutput->points.size() << std::endl;
  	pcl::io::savePLYFile(outFilename, *cloudOutput);
	}
	else if (toolname == "xxx")
	{
		int modelType = getModelType(argv[argn++]);
		float threshold = atof(argv[argn++]);
		int minInliers = atoi(argv[argn++]);
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

  	pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

  	pcl::io::loadPLYFile(inFilename, *cloudInput);
  	std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		for (unsigned i = 0;; i++)
  	{
			std::pair<pcl::PointIndices::Ptr, pcl::ModelCoefficients::Ptr> pairIndicesModel = sacSegmentation(cloudInput, modelType, threshold);

			if (pairIndicesModel.first->indices.size() < minInliers)
			{
				std::cerr << "Too few inliers: " << pairIndicesModel.first->indices.size() << std::endl;
				break;
			}

			Cloud::Ptr cloudInliers = filterByIndices(cloudInput, pairIndicesModel.first, false);	
			Cloud::Ptr cloudOutliers = filterByIndices(cloudInput, pairIndicesModel.first, true);	
    	cloudInput.swap(cloudOutliers);

			Cloud::Ptr cloudOutput = projectInliers(cloudInliers, modelType, pairIndicesModel.second);

	  	std::cerr << "Index: " << i << " Output Cloud Size: " << cloudOutput->points.size() << std::endl;
  		pcl::io::savePLYFile(createFilename(outFilename, i), *cloudOutput);
		}
	}
	
  return (0);
}

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


float distance(Point p1, Point p2)
{
    return ::sqrt(::pow((p1.x-p2.x), 2) + ::pow((p1.y-p2.y), 2) + ::pow((p1.z-p2.z), 2));
}

std::pair<int, int> get_min_max(Cloud::Ptr cloud)
{
    int max_i = 0;
    int max_j = 0;
    float max_dist = 0;
    for (unsigned i = 0; i < cloud->points.size()-1; ++i)
        for (unsigned j = i; j < cloud->points.size()-1; ++j)
        {
            if (distance(cloud->points[i],  cloud->points[j]) > max_dist)
            {
                max_i = i;
                max_j = j;
                max_dist = distance(cloud->points[i],  cloud->points[j]);
            }
        }

    return std::pair<int, int>(max_i,max_j);
}

void saveLines(std::vector<Point> lines)
{
    printf("ply\n");
    printf("format ascii 1.0\n");
    printf("element vertex %d\n", lines.size()*2);
    printf("property float x\n");
    printf("property float y\n");
    printf("property float z\n");
    printf("element line %d\n", lines.size());
    printf("property vertex_index\n");
    printf("end_header\n");

    for (unsigned n = 0; n < lines.size(); n+=2)
    {
        printf("%f %f %f\n",
            lines[n].x,
            lines[n].y,
            lines[n].z);
        printf("%f %f %f\n",
            lines[n+1].x,
            lines[n+1].y,
            lines[n+1].z);

    }

    for (unsigned n = 0; n < lines.size(); ++n)
    {
        printf("%d %d\n", 2*n, 2*n+1);
    }
}


std::vector<float> getMinmax(Cloud::Ptr cloudInput)
{
	std::vector<float> vecOutput;
	vecOutput.push_back(std::numeric_limits<float>::max());
	vecOutput.push_back(std::numeric_limits<float>::max());
	vecOutput.push_back(std::numeric_limits<float>::max());
	vecOutput.push_back(std::numeric_limits<float>::min());
	vecOutput.push_back(std::numeric_limits<float>::min());
	vecOutput.push_back(std::numeric_limits<float>::min());

	for (unsigned i = 0; i < cloudInput->points.size(); ++i)
	{
		if (cloudInput->points[i].x < vecOutput[0])
			vecOutput[0] = cloudInput->points[i].x;
		if (cloudInput->points[i].y < vecOutput[1])
			vecOutput[1] = cloudInput->points[i].y;
		if (cloudInput->points[i].z < vecOutput[2])
			vecOutput[2] = cloudInput->points[i].z;

		if (cloudInput->points[i].x > vecOutput[3])
			vecOutput[3] = cloudInput->points[i].x;
		if (cloudInput->points[i].y > vecOutput[4])
			vecOutput[4] = cloudInput->points[i].y;
		if (cloudInput->points[i].z > vecOutput[5])
			vecOutput[5] = cloudInput->points[i].z;
	}

	return vecOutput;
}

Cloud::Ptr merge(Cloud::Ptr cloudOutput, Cloud::Ptr cloudInput)
{
	for (unsigned i = 0; i < cloudInput->points.size(); ++i)
	{
		cloudOutput->points.push_back(cloudInput->points[i]);
	}

	//cloudOutput.size(cloudOutput.size() + cloudInput.size());

	return cloudOutput;
}

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

	if (toolname == "get-minmax")
	{
		const char* inFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		std::vector<float> minmax = getMinmax(cloudInput);
	}

	else if (toolname == "pass-through")
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

	else if (toolname == "slice")
	{
		int numSlices = atoi(argv[argn++]);
		const char* fieldName = argv[argn++];
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		std::vector<float> minmax = getMinmax(cloudInput);

		unsigned fieldMinIndex = std::string(fieldName) == "x" ? 0 : (std::string(fieldName) == "y" ? 1 : 2);
		unsigned fieldMaxIndex = fieldMinIndex + 3;

		for (unsigned i = 0; i < numSlices; ++i)
		{
			float min = minmax[fieldMinIndex] + (minmax[fieldMaxIndex]-minmax[fieldMinIndex])/numSlices * (i+0);
			float max = minmax[fieldMinIndex] + (minmax[fieldMaxIndex]-minmax[fieldMinIndex])/numSlices * (i+1);
			Cloud::Ptr cloudOutput = passThrough(cloudInput, fieldName, min, max);

			std::cerr << "Index: " << i << " Output Cloud Size: " << cloudOutput->points.size() << std::endl;
			pcl::io::savePLYFile(createFilename(outFilename, i), *cloudOutput);
		}
	}

	else if (toolname == "merge")
	{
		std::vector<std::string> inFilenameList;
		while (argn < (argc-1))
			inFilenameList.push_back(argv[argn++]);
		const char* outFilename = argv[argn++];

		Cloud::Ptr cloudOutput(new Cloud);
		for (unsigned i = 0; i < inFilenameList.size(); ++i)
		{
			pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

			pcl::io::loadPLYFile(inFilenameList[i], *cloudInput);
			std::cerr << "Index: " << i << " Input Cloud Size: " << cloudInput->points.size() << std::endl;

			cloudOutput = merge(cloudOutput, cloudInput);
		}

		std::cerr << " Output Cloud Size: " << cloudOutput->points.size() << std::endl;
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

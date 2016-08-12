#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/PolygonMesh.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
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
#include <pcl/segmentation/region_growing.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point> Cloud;

Cloud::Ptr toCloud(std::vector<Point> pointList)
{
	Cloud::Ptr out(new Cloud);

	for (unsigned i = 0; i < pointList.size(); ++i)
		out->points.push_back(pointList[i]);

	return out;
}

std::vector<Point> fromCloud(Cloud::Ptr cloud)
{
	std::vector<Point> out;

  for (unsigned i = 0; i < cloud->points.size(); ++i)
    out.push_back(cloud->points[i]);

	return out;
}

Eigen::Vector3f toVec(Point in)
{
	Eigen::Vector3f out;
  
	out(0) = in.x;
  out(1) = in.y;
  out(2) = in.z;

	return out;
}

Point fromVec(Eigen::Vector3f in)
{
	return Point(in(0), in(1), in(2));
}


std::vector<Eigen::Vector3f> toVec(std::vector<Point> in)
{
	std::vector<Eigen::Vector3f> out;

	for (unsigned i = 0; i < in.size(); ++i)
		out.push_back(toVec(in[i]));

	return out;
}

std::vector<Point> fromVec(std::vector<Eigen::Vector3f> in)
{
	std::vector<Point> out;

	for (unsigned i = 0; i < in.size(); ++i)
		out.push_back(fromVec(in[i]));

	return out;
}

pcl::PolygonMesh::Ptr extractFaces(Cloud::Ptr cloudInput)
{
	std::vector<Eigen::Vector3f> directions;

	for (unsigned i = 0; i < cloudInput->points.size(); i+=2)
	{
		Eigen::Vector3f p1 = toVec(cloudInput->points[i]);
		Eigen::Vector3f p2 = toVec(cloudInput->points[i+1]);
		
		Eigen::Vector3f v = p1-p2;
		v = v/::sqrt(v.dot(v));
		directions.push_back(v);
	}

	pcl::PolygonMesh::Ptr meshOutput(new pcl::PolygonMesh);
	pcl::toPCLPointCloud2(*cloudInput, meshOutput->cloud);

	for (unsigned i = 0; i < directions.size(); ++i)
	{
		for (unsigned j = i+1; j < directions.size(); ++j)
		{
			Eigen::Vector3f v = directions[i].cross(directions[j]); 

			if (::abs(v.dot(v)) < 0.05)
			{
				if (directions[i].dot(directions[j]) < 0)
				{
					pcl::Vertices polygon;
					polygon.vertices.push_back(i*2);
					polygon.vertices.push_back(i*2+1);
					polygon.vertices.push_back(j*2);
					polygon.vertices.push_back(j*2+1);

					meshOutput->polygons.push_back(polygon);
				}
				else
				{
					pcl::Vertices polygon;
					polygon.vertices.push_back(i*2);
					polygon.vertices.push_back(i*2+1);
					polygon.vertices.push_back(j*2+1);
					polygon.vertices.push_back(j*2);

					meshOutput->polygons.push_back(polygon);
				}
			}
		}
	}

	return meshOutput;
}

float distance(Point p1, Point p2)
{
    return ::sqrt(::pow((p1.x-p2.x), 2) + ::pow((p1.y-p2.y), 2) + ::pow((p1.z-p2.z), 2));
}

float distance(Point point, std::vector<Point> listPoints)
{
	float out = 0;

	for (unsigned i = 0; i < listPoints.size(); ++i)
		out += distance(point, listPoints[i]);

	return out;
} 

Cloud::Ptr getMaxDistance(Cloud::Ptr cloudInput)
{
	int max_i = 0;
	int max_j = 0;
	float max_dist = 0;
	for (unsigned i = 0; i < cloudInput->points.size()-1; ++i)
		for (unsigned j = i; j < cloudInput->points.size()-1; ++j)
		{
			if (distance(cloudInput->points[i],  cloudInput->points[j]) > max_dist)
			{
				max_i = i;
				max_j = j;
				max_dist = distance(cloudInput->points[i],  cloudInput->points[j]);
			}
		}

	Cloud::Ptr cloudOutput(new Cloud);
	cloudOutput->points.push_back(cloudInput->points[max_i]);
	cloudOutput->points.push_back(cloudInput->points[max_j]);

	return cloudOutput;
}

void savePLYFile(std::string filename, Cloud::Ptr cloudInput)
{
	pcl::PLYWriter writer;
	writer.write<Point>(filename, *cloudInput, false, false);
}

void savePLYEdges(std::string fileName, Cloud::Ptr cloudInput)
{
	std::ofstream file;
  	file.open(fileName.c_str());

	file << "ply\n";
	file << "format ascii 1.0\n";
	file << "element vertex " << cloudInput->points.size() << "\n";
	file << "property float x\n";
	file << "property float y\n";
	file << "property float z\n";
	file << "element edge " << cloudInput->points.size()/2 << "\n";
	file << "property int vertex1\n";
	file << "property int vertex2\n";
	file << "end_header\n";

	for (unsigned n = 0; n < cloudInput->points.size(); n+=2)
	{
		file << cloudInput->points[n].x << " ";
		file << cloudInput->points[n].y << " ";
		file << cloudInput->points[n].z << "\n";

		file << cloudInput->points[n+1].x << " ";
		file << cloudInput->points[n+1].y << " ";
		file << cloudInput->points[n+1].z << "\n";
	}

	for (unsigned n = 0; n < cloudInput->points.size(); n+=2)
	{
		file << n << " " << n+1 << "\n";
	}

	file.close();
}

//----------------------------------------------------------------
pcl::Vertices createPolygon(unsigned i0, unsigned i1, unsigned i2, unsigned i3)
{
	pcl::Vertices polygon;

	polygon.vertices.push_back(i0);
	polygon.vertices.push_back(i1);
	polygon.vertices.push_back(i2);
	polygon.vertices.push_back(i3);

	return polygon;
}

std::vector<Point> getPlane(Point p1, Point p2, Point p3)
{
	std::vector<Point> out;

	out.push_back(p1);
	out.push_back(fromVec((toVec(p2) - toVec(p1)).cross(toVec(p3) - toVec(p1))));

	return out;
}

std::vector<Point> getLine(Point p1, Point p2)
{
	std::vector<Point> out;

	out.push_back(p1);
	out.push_back(fromVec(toVec(p2) - toVec(p1)));

	return out;
}

Eigen::Vector3f cutWithPlane(std::vector<Eigen::Vector3f> line, std::vector<Eigen::Vector3f> plane)
{
	float x = (-((line[0] - plane[0]).dot(plane[1]))/(line[1]).dot(plane[1]));
	return line[1] * x + line[0];
}

std::vector< std::vector<Point> > getFaces(std::vector<Point> box)
{
	std::vector< std::vector<Point> > out;

	std::vector<pcl::Vertices> polygons;
	polygons.push_back(createPolygon(0, 1, 2, 3));
	polygons.push_back(createPolygon(4, 5, 6, 7));
	polygons.push_back(createPolygon(0, 2, 4, 6));
	polygons.push_back(createPolygon(1, 3, 5, 7));
	polygons.push_back(createPolygon(0, 1, 4, 5));
	polygons.push_back(createPolygon(2, 3, 6, 7));
	
	for (unsigned i = 0; i < polygons.size(); ++i)
	{
		std::vector<Point> face;
		for (unsigned j = 0; j < polygons[i].vertices.size(); ++j)
		{
			face.push_back(box[polygons[i].vertices[j]]);
		}

		out.push_back(face);
	}

	return out;
}

std::vector<Point> alignBox(std::vector<Point> boxRef, std::vector<Point> boxTgt)
{
	std::vector< std::vector<Point> > faces = getFaces(boxRef);
	
	unsigned minIdx = 0;
	float minDist = std::numeric_limits<float>::max();
	// exclude bottom plane
	for (unsigned i = 1; i < faces.size(); ++i)
	{
		if (distance(boxTgt[0], faces[i]) < minDist)
		{
			minIdx = i;
			minDist = distance(boxTgt[0], faces[i]);
		}
	}

	// extend box to align with plane
	std::vector<Point> boxOut(boxTgt.size());
	boxOut[0] = boxTgt[0];
	boxOut[1] = boxTgt[1];
	boxOut[2] = boxTgt[2];
	boxOut[3] = boxTgt[3];
	boxOut[4] = fromVec(cutWithPlane(toVec(getLine(boxTgt[0], boxTgt[4])), toVec(getPlane(faces[minIdx][0], faces[minIdx][1], faces[minIdx][2]))));
	boxOut[5] = fromVec(cutWithPlane(toVec(getLine(boxTgt[1], boxTgt[5])), toVec(getPlane(faces[minIdx][0], faces[minIdx][1], faces[minIdx][2]))));
	boxOut[6] = fromVec(cutWithPlane(toVec(getLine(boxTgt[2], boxTgt[6])), toVec(getPlane(faces[minIdx][0], faces[minIdx][1], faces[minIdx][2]))));
	boxOut[7] = fromVec(cutWithPlane(toVec(getLine(boxTgt[3], boxTgt[7])), toVec(getPlane(faces[minIdx][0], faces[minIdx][1], faces[minIdx][2]))));

	return boxOut;
}

std::vector<Point> createConvexHull(Point min, Point max)
{
	std::vector<Point> out;

	out.push_back(min);
	out.push_back(Point(max.x, min.y, min.z)); 
	out.push_back(Point(min.x, max.y, min.z)); 
	out.push_back(Point(max.x, max.y, min.z)); 
	out.push_back(Point(min.x, min.y, max.z)); 
	out.push_back(Point(max.x, min.y, max.z)); 
	out.push_back(Point(min.x, max.y, max.z)); 
	out.push_back(max);

	return out;
}

//----------------------------------------------------------------

Cloud::Ptr getMinMax(Cloud::Ptr cloudInput)
{
	Cloud::Ptr cloudOutput(new Cloud);

	Point min;
	Point max;

	pcl::getMinMax3D(*cloudInput, min, max);

	cloudOutput->points.push_back(min);
	cloudOutput->points.push_back(max);

	return cloudOutput;
}

// create surface for bounding box
pcl::PolygonMesh::Ptr makeSurface(Cloud::Ptr cloudInput)
{
	pcl::PolygonMesh::Ptr meshOutput(new pcl::PolygonMesh);
	pcl::toPCLPointCloud2(*cloudInput, meshOutput->cloud);

	meshOutput->polygons.push_back(createPolygon(0, 1, 2, 3));
	meshOutput->polygons.push_back(createPolygon(4, 5, 6, 7));
	meshOutput->polygons.push_back(createPolygon(0, 2, 4, 6));
	meshOutput->polygons.push_back(createPolygon(1, 3, 5, 7));
	meshOutput->polygons.push_back(createPolygon(0, 1, 4, 5));
	meshOutput->polygons.push_back(createPolygon(2, 3, 6, 7));

	return meshOutput;	
}

#include <algorithm>

bool comparePointOnX(Point p1, Point p2)
{ return p1.x < p2.x; }
bool comparePointOnY(Point p1, Point p2)
{ return p1.y < p2.y; }
bool comparePointOnZ(Point p1, Point p2)
{ return p1.z < p2.z; }

Cloud::Ptr getBoundingBox(Cloud::Ptr cloud)
{
	Cloud proj; 
	pcl::PCA< Point > pca; 
	pca.setInputCloud(cloud); 
	pca.project(*cloud, proj); 

	Point proj_min; 
	Point proj_max; 
	pcl::getMinMax3D(proj, proj_min, proj_max);

	std::vector<Point> projPointList = createConvexHull(proj_min, proj_max);

	std::vector<Point> pointList(projPointList.size());
	for (unsigned i = 0; i < projPointList.size(); ++i) 
		pca.reconstruct(projPointList[i], pointList[i]); 

	std::sort(pointList.begin()+0, pointList.begin()+8, comparePointOnZ);
	std::sort(pointList.begin()+0, pointList.begin()+4, comparePointOnY);
	std::sort(pointList.begin()+4, pointList.begin()+8, comparePointOnY);
	std::sort(pointList.begin()+0, pointList.begin()+2, comparePointOnX);
	std::sort(pointList.begin()+2, pointList.begin()+4, comparePointOnX);
	std::sort(pointList.begin()+4, pointList.begin()+6, comparePointOnX);
	std::sort(pointList.begin()+6, pointList.begin()+8, comparePointOnX);

	return toCloud(pointList);
}

#if 0
Cloud::Ptr getBoundingBox(Cloud::Ptr cloud)
{
	Eigen::Vector4f pcaCentroid;
	pcl::compute3DCentroid(*cloud, pcaCentroid);
	Eigen::Matrix3f covariance;
	computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(covariance, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigenVectorsPCA = eigenSolver.eigenvectors();
	/// This line is necessary for proper orientation in some cases. The numbers come out the same without it, but
	/// the signs are different and the box doesn't get correctly oriented in some cases.
	eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));

	// Transform the original cloud to the origin where the principal components correspond to the axes.
	Cloud::Ptr cloudProjected(new Cloud);

	Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
	projectionTransform.block<3,3>(0,0) = eigenVectorsPCA.transpose();
	projectionTransform.block<3,1>(0,3) = -1.f * (projectionTransform.block<3,3>(0,0) * pcaCentroid.head<3>());
	pcl::transformPointCloud(*cloud, *cloudProjected, projectionTransform);
	Point minPoint, maxPoint;
	pcl::getMinMax3D(*cloudProjected, minPoint, maxPoint);

	// rotation and translation
	const Eigen::Quaternionf bboxQuaternion(eigenVectorsPCA);
	const Eigen::Vector3f meanDiagonal = 0.5f*(maxPoint.getVector3fMap() + minPoint.getVector3fMap());
	const Eigen::Vector3f bboxTransform = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();

	vtkSmartPointer<vtkDataSet> cube = pcl::visualization::createCube(bboxTransform, bboxQuaternion, maxPoint.x - minPoint.x, maxPoint.y - minPoint.y, maxPoint.z - minPoint.z);


	Cloud::Ptr cloudOutput(new Cloud);

	for (int i = 0; i < cube->GetNumberOfPoints() and i < 8; i+=1)
		cloudOutput->points.push_back(Point(cube->GetPoint(i)[0], cube->GetPoint(i)[1], cube->GetPoint(i)[2]));

	return cloudOutput;
}
#endif

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
	*cloudOutput += *cloudInput;
	return cloudOutput;
}

pcl::PolygonMesh::Ptr merge(pcl::PolygonMesh::Ptr meshOutput, pcl::PolygonMesh::Ptr meshInput)
{
	Cloud cloudOutput;
	Cloud cloudInput;

	pcl::fromPCLPointCloud2(meshOutput->cloud, cloudOutput);
	pcl::fromPCLPointCloud2(meshInput->cloud, cloudInput);

	unsigned offset = cloudOutput.points.size();
	cloudOutput += cloudInput;

	pcl::toPCLPointCloud2(cloudOutput, meshOutput->cloud);

	for (unsigned i = 0; i < meshInput->polygons.size(); ++i)
	{
		pcl::Vertices polygon;
		for (unsigned j = 0; j < meshInput->polygons[i].vertices.size(); ++j)
		{
				polygon.vertices.push_back(meshInput->polygons[i].vertices[j]+offset);
		}

		meshOutput->polygons.push_back(polygon);
	}

	return meshOutput;
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

pcl::PointCloud<pcl::Normal>::Ptr estimateNormals(Cloud::Ptr cloudInput, float radius)
{
	pcl::PointCloud<pcl::Normal>::Ptr cloudOutput(new pcl::PointCloud<pcl::Normal>);

	pcl::search::KdTree<Point>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());

	pcl::NormalEstimation<Point, pcl::Normal> tool;
	tool.setInputCloud(cloudInput);
	tool.setSearchMethod(tree);
	tool.setRadiusSearch(radius);
	tool.compute(*cloudOutput);

	return cloudOutput;
}

std::vector<Cloud::Ptr> euclidianClustering(Cloud::Ptr cloudInput, float clusterTolerance, int minClusterSize, int maxClusterSize)
{
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

	std::vector<Cloud::Ptr> listCloudOutput;

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

std::vector<Cloud::Ptr> regionGrowing(Cloud::Ptr cloudInput, pcl::PointCloud <pcl::Normal>::Ptr cloudNormal, int numberOfNeighbours, int minClusterSize, int maxClusterSize)
{
	pcl::search::KdTree<Point>::Ptr tree(new pcl::search::KdTree<Point>);
	tree->setInputCloud(cloudInput);

	std::vector <pcl::PointIndices> clusterIndices;
	pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> tool;
	tool.setSearchMethod(tree);
	tool.setNumberOfNeighbours(numberOfNeighbours);
	tool.setMinClusterSize(minClusterSize);
	tool.setMaxClusterSize(maxClusterSize);
	tool.setInputCloud(cloudInput);
	//tool.setIndices (indices);
	tool.setInputNormals(cloudNormal);
	tool.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
	tool.setCurvatureThreshold(1.0);
	tool.extract(clusterIndices);

	std::vector<Cloud::Ptr> listCloudOutput;

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

	pcl::ExtractIndices<Point> tool;
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

	else if (toolname == "make-surface")
	{
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		pcl::PolygonMesh::Ptr meshOutput = makeSurface(cloudInput);

		std::cerr << "Output Mesh Size: " << meshOutput->polygons.size() << std::endl;
		pcl::io::savePLYFile(outFilename, *meshOutput);
	}

	else if (toolname == "get-bounding-box")
	{
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		Cloud::Ptr cloudOutput = getBoundingBox(cloudInput);

		std::cerr << "Output Cloud: " << outFilename << " Size: " << cloudOutput->points.size() << std::endl;
		pcl::io::savePLYFile(outFilename, *cloudOutput);
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

	else if (toolname == "min-max")
	{
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		Cloud::Ptr cloudOutput = getMinMax(cloudInput);

		std::cerr << "Output Cloud Size: " << cloudOutput->points.size() << std::endl;
		pcl::io::savePLYFile(outFilename, *cloudOutput);
	}

	else if (toolname == "concave-hull")
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

	
	else if (toolname == "region-growing")
	{
		int numberOfNeighbours = atoi(argv[argn++]);
		int minClusterSize = atoi(argv[argn++]);
		int maxClusterSize = atoi(argv[argn++]);
		const char* inFilename = argv[argn++];
		const char* normalFilename = argv[argn++];
		const char* outFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		pcl::PointCloud<pcl::Normal>::Ptr cloudNormal(new pcl::PointCloud<pcl::Normal>);

		cloudNormal = estimateNormals(cloudInput, 0.1);
		//pcl::io::loadPLYFile(inFilename, *cloudNormal);
		//std::cerr << "Normal Cloud Size: " << cloudNormal->points.size() << std::endl;

		std::vector<Cloud::Ptr> listCloudOutput = regionGrowing(cloudInput, cloudNormal, numberOfNeighbours, minClusterSize, maxClusterSize);

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
		float start = atof(argv[argn++]);
		float end = atof(argv[argn++]);
		int numSlices = atoi(argv[argn++]);
		const char* fieldName = argv[argn++];
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		for (unsigned i = 0; i < numSlices; ++i)
		{
			float min = start + (end-start)/numSlices * (i+0);
			float max = start + (end-start)/numSlices * (i+1);
			Cloud::Ptr cloudOutput = passThrough(cloudInput, fieldName, min, max);

			std::cerr << "Index: " << i << " Output Cloud Size: " << cloudOutput->points.size() << std::endl;
			savePLYFile(createFilename(outFilename, i), cloudOutput);
		}
	}

	else if (toolname == "merge-cloud")
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

		std::cerr << "Output Cloud Size: " << cloudOutput->points.size() << std::endl;
		pcl::io::savePLYFile(outFilename, *cloudOutput);
	}

	else if (toolname == "merge-mesh")
	{
		std::vector<std::string> inFilenameList;
		while (argn < (argc-1))
			inFilenameList.push_back(argv[argn++]);
		const char* outFilename = argv[argn++];

		pcl::PolygonMesh::Ptr meshOutput(new pcl::PolygonMesh);
		for (unsigned i = 0; i < inFilenameList.size(); ++i)
		{
			pcl::PolygonMesh::Ptr meshInput(new pcl::PolygonMesh);

			pcl::io::loadPLYFile(inFilenameList[i], *meshInput);
			std::cerr << "Index: " << i << " Input Cloud Size: " << meshInput->polygons.size() << std::endl;

			meshOutput = merge(meshOutput, meshInput);
		}

		std::cerr << "Output Mesh Size: " << meshOutput->polygons.size() << std::endl;
		pcl::io::savePLYFile(outFilename, *meshOutput);
	}

	else if (toolname == "model-segment")
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

	else if (toolname == "get-max-distance")
	{
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		Cloud::Ptr cloudOutput = getMaxDistance(cloudInput);

		std::cerr << "Output Cloud Size: " << cloudOutput->points.size() << std::endl;
		pcl::io::savePLYFile(outFilename, *cloudOutput);
	}

	else if (toolname == "extract-faces")
	{
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		pcl::PolygonMesh::Ptr meshOutput = extractFaces(cloudInput);

		std::cerr << "Output Mesh Size: " << meshOutput->polygons.size() << std::endl;
		pcl::io::savePLYFile(outFilename, *meshOutput);
	}

	else if (toolname == "estimate-normals")
	{
		float radius = atof(argv[argn++]);
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		pcl::PointCloud<pcl::Normal>::Ptr cloudOutput = estimateNormals(cloudInput, radius);

		std::cerr << "Output Cloud Size: " << cloudOutput->points.size() << std::endl;
		pcl::io::savePLYFile(outFilename, *cloudOutput);
	}

	else if (toolname == "create-edges")
	{
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		// nothing to do here
		Cloud::Ptr cloudOutput = cloudInput;

		std::cerr << "Output Cloud Size: " << cloudOutput->points.size() << std::endl;
		savePLYEdges(outFilename, cloudOutput);
	}

	else if (toolname == "create-mesh")
	{
		std::vector<std::string> inFilenameList;
		while (argn < (argc-1))
			inFilenameList.push_back(argv[argn++]);
		const char* outFilename = argv[argn++];

		pcl::PolygonMesh::Ptr meshOutput(new pcl::PolygonMesh);
		for (unsigned i = 0; i < inFilenameList.size(); ++i)
		{
			pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

			pcl::io::loadPLYFile(inFilenameList[i], *cloudInput);
			std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

			meshOutput = merge(meshOutput, extractFaces(getMinMax(cloudInput)));
			std::cerr << "output Mesh Size: " << meshOutput->polygons.size() << std::endl;
		}

		std::cerr << "Output Mesh Size: " << meshOutput->polygons.size() << std::endl;
		pcl::io::savePLYFile(outFilename, *meshOutput);
	}

	else if (toolname == "xxx")
	{
		float radian = atof(argv[argn++]);
		const char* refFilename = argv[argn++];
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

#if 0
		std::vector<Point> boxRef = createConvexHull(Point(0,0,0), Point(1,3,10));
		std::vector<Point> boxIn = createConvexHull(Point(5,1,0), Point(6,2,5));

		Cloud::Ptr cloudIn(new Cloud);
		Cloud::Ptr cloudTransf(new Cloud);
		cloudIn = toCloud(boxIn);
		//Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
		Eigen::Affine3f transform = Eigen::Affine3f::Identity();
		transform.rotate(Eigen::AngleAxisf(radian, Eigen::Vector3f::UnitY()));
		pcl::transformPointCloud(*cloudIn, *cloudTransf, transform);
		boxIn = fromCloud(cloudTransf);
#endif

		pcl::PointCloud<Point>::Ptr cloudRef(new Cloud);

		pcl::io::loadPLYFile(refFilename, *cloudRef);
		std::cerr << "Reference Cloud Size: " << cloudRef->points.size() << std::endl;
		
		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;
		
		std::vector<Point> boxRef = fromCloud(cloudRef);
		std::vector<Point> boxIn = fromCloud(cloudInput);

		std::vector<Point> boxOut = alignBox(boxRef, boxIn);

		pcl::PolygonMesh::Ptr meshOutput = makeSurface(toCloud(boxOut)); 

    std::cerr << "Output Mesh: " << outFilename << " Size: " << meshOutput->polygons.size() << std::endl;
    pcl::io::savePLYFile(outFilename, *meshOutput);
	}

	else
	{
		std::cerr << "Unknown toolname:" << toolname << "\n";
	}


	return (0);
}

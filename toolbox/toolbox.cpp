#include <cfloat>
#include <cmath>
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
#include <pcl/surface/convex_hull.h>
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

Point fromVec(Eigen::Vector4f in)
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

//----------------------------------------------------------------
// geometry helper

float sqr(Eigen::Vector3f v)
{
	return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

float length(Eigen::Vector3f v)
{
  return ::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

bool isWithin(Point p, Point min, Point max)
{
	return (p.x >= min.x and p.x <= max.x) and (p.y >= min.y and p.y <= max.y);
}

Point addxy(Point p, Eigen::Vector3f v)
{
	return Point(p.x+v(0), p.y+v(1), p.z);
}

Point addx(Point p, Eigen::Vector3f v)
{
	return Point(p.x+v(0), p.y, p.z);
}

Point addy(Point p, Eigen::Vector3f v)
{
	return Point(p.x, p.y+v(1), p.z);
}


struct Line
{
	Line()
	: p(Eigen::Vector3f::Zero()), d(Eigen::Vector3f::Zero())
	{}

	Line(Point p1, Point p2)
	{
		p = toVec(p1);
		d = Eigen::Vector3f(toVec(p2) - toVec(p1));
	}

	Line(Eigen::Vector3f p0, Eigen::Vector3f p1)
	{
		p = p0;
		d = p1-p0;
	}

	float distance(Point point)
	{
		return length((toVec(point)-p).cross(d))/length(d);
	}

	Eigen::Vector3f p;
	Eigen::Vector3f d;
};


struct Plane
{
	Plane()
	: p(), n(), d(0)
	{}

	Plane(Point p1, Point p2, Point p3)
	{
		p = toVec(p1);
		n = (toVec(p2) - toVec(p1)).cross(toVec(p3) - toVec(p1));
		n = n/length(n);
		if (toVec(p1).dot(n) < 0)
			n = -n;
		d = n.dot(toVec(p1));
	}

	Line cut(Plane plane)
	{
		Line line;

		float b = sqr(n)*sqr(plane.n)-n.dot(plane.n)*n.dot(plane.n);
		line.p = ((d*sqr(plane.n) - plane.d*(n.dot(plane.n)))*n)/b + ((plane.d*sqr(n) - d*(n.dot(plane.n)))*plane.n)/b;
		line.d = n.cross(plane.n);

		return line;
	}

	Point cut(Line line)
	{
		float x = (-((line.p - p).dot(n))/line.d.dot(n));
		return fromVec(Eigen::Vector3f(line.d * x + line.p));
	}

	Eigen::Vector3f p;
	Eigen::Vector3f n;
	float d;
};

float distPointToLine(Point point, Line line)
{
	return length((toVec(point) - line.p).cross(line.d)) / length(line.d);
}

float distPointToPlane(Point point, Plane plane)
{
//	return (toVec(point) - plane.p).dot(plane.n) / length(plane.n);
	return toVec(point).dot(plane.n) - plane.d;
}

float getMinDistToPlanes(Point point, std::vector<Plane> planes)
{
  float minDist = FLT_MAX;

  for (unsigned i = 0; i < planes.size(); ++i)
	{
    float d = fabs(distPointToPlane(point, planes[i]));
    if (d < minDist)
      minDist = d;
	}

  return minDist;
}

float getMinDistToRect(Point point, std::vector<Eigen::Vector3f> rect)
{
  float minDist = FLT_MAX;

  std::vector<Line> l;
  l.push_back(Line(rect[0], rect[1]));
  l.push_back(Line(rect[1], rect[2]));
  l.push_back(Line(rect[2], rect[3]));
  l.push_back(Line(rect[3], rect[0]));

  for (unsigned i = 0; i < l.size(); ++i)
	{
    float d = distPointToLine(point, l[i]);
    if (d < minDist)
      minDist = d;
	}

  return minDist;
}

Eigen::Vector3f getMax(Eigen::Vector3f v1, Eigen::Vector3f v2)
{
	return length(v1) > length(v2) ? v1 : v2;
}

float volume(Point p1 , Point p2)
{
	return (p2.x-p1.x)*(p2.y-p1.y)*(p2.z-p1.z);
}

//----------------------------------------------------------------
// central datastructure for reconstruction

std::vector<Point> createConvexHull(Point min, Point max);

struct Cluster
{
		unsigned sliceId;
		Cloud::Ptr cloud;
		Cloud::Ptr minmax;
		Cloud::Ptr box;

		float volume()
		{
			return ::volume(minmax->points[0], minmax->points[1]);
		}

		float volumeBox()
		{
			return ::volume(box->points[0], box->points[7]);
		}

		bool overlapsWith(const Cluster& o)
		{
			return ((minmax->points[0].x >= o.minmax->points[0].x and minmax->points[0].x <= o.minmax->points[1].x) or
					 (minmax->points[1].x >= o.minmax->points[0].x and minmax->points[1].x <= o.minmax->points[1].x) or
					 (minmax->points[0].x >= o.minmax->points[0].x and minmax->points[1].x <= o.minmax->points[1].x) or
					 (minmax->points[0].x <= o.minmax->points[0].x and minmax->points[1].x >= o.minmax->points[1].x)) and
				((minmax->points[0].y >= o.minmax->points[0].y and minmax->points[0].y <= o.minmax->points[1].y) or
					 (minmax->points[1].y >= o.minmax->points[0].y and minmax->points[1].y <= o.minmax->points[1].y) or
					 (minmax->points[0].y >= o.minmax->points[0].y and minmax->points[1].y <= o.minmax->points[1].y) or
					 (minmax->points[0].y <= o.minmax->points[0].y and minmax->points[1].y >= o.minmax->points[1].y));

		}

		Cloud::Ptr moveBox(Cluster& cluster)
		{
			Cloud::Ptr clusterBox(new Cloud);

			Eigen::Vector3f d1 = toVec(minmax->points[0]) - toVec(cluster.minmax->points[0]);
			Eigen::Vector3f d2 = toVec(minmax->points[1]) - toVec(cluster.minmax->points[1]);

			Eigen::Vector3f d = (d1+d2)/2;

			for (unsigned i = 0; i < box->points.size(); ++i)
			{
				Eigen::Vector3f v = toVec(box->points[i])-d;
				clusterBox->points.push_back(fromVec(v));
			}

			return clusterBox;
		}
		
	std::vector<Plane> getPlanes()
	{
  	std::vector<Plane> planes;
		planes.push_back(Plane(box->points[0], box->points[1], box->points[4]));
		planes.push_back(Plane(box->points[1], box->points[2], box->points[5]));
		planes.push_back(Plane(box->points[2], box->points[3], box->points[6]));
		planes.push_back(Plane(box->points[3], box->points[0], box->points[7]));

		return planes;
	}

	float getMeanDist()
	{
		std::vector<Plane> planes = getPlanes();

		float dist = 0;
		for (unsigned i = 0; i < cloud->points.size(); ++i)
		{
			dist += getMinDistToPlanes(cloud->points[i], planes);
		}
		dist /= cloud->points.size();

		return dist;
	}

	void approximateSides(Eigen::Vector3f d)
	{
		std::vector<Point> bbox = createConvexHull(minmax->points[0], minmax->points[1]);

		for (unsigned i = 0; i < 4; ++i)
		{
			Point p = bbox[i];

			if (isWithin(addxy(p, d), minmax->points[0], minmax->points[1]))
				box->points.push_back(addxy(p, d));
			else if (isWithin(addx(p, d), minmax->points[0], minmax->points[1]))
				box->points.push_back(addx(p, d));
			else if (isWithin(addy(p, d), minmax->points[0], minmax->points[1]))
				box->points.push_back(addy(p, d));
			else
				box->points.push_back(p);
		}

		d = -d;
		for (unsigned i = 4; i < 8; ++i)
		{
			Point p = bbox[i];

			if (isWithin(addxy(p, d), minmax->points[0], minmax->points[1]))
			{
				box->points.push_back(addxy(p, d));
				printf("addxy\n");
			}
			else if (isWithin(addx(p, d), minmax->points[0], minmax->points[1]))
			{
				box->points.push_back(addx(p, d));
				printf("addx\n");
			}
			else if (isWithin(addy(p, d), minmax->points[0], minmax->points[1]))
			{
				box->points.push_back(addy(p, d));
				printf("addy\n");
			}
			else
				box->points.push_back(p);
		}
	}
};

struct Bar
{
	std::vector<unsigned> listCluster;
	Line lineCenter;
};

//----------------------------------------------------------------

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

struct Color
{
	Color()
	: red(255), green(255), blue(255)
	{}

	Color(unsigned red_, unsigned green_, unsigned blue_)
	: red(red_), green(green_), blue(blue_)
	{}

	unsigned red;
	unsigned green;
	unsigned blue;
};

struct Edge
{
	Edge()
	: start(), end(), color()
	{}

	Edge(unsigned start_, unsigned end_)
	: start(start_), end(end_)
	{}

	Edge(unsigned start_, unsigned end_, Color color_)
	: start(start_), end(end_), color(color_)
	{}

	unsigned start;
	unsigned end;
	Color color;
};

typedef std::vector<Edge> Edges;

Edges createEdges(unsigned numEdges)
{
	Edges edges;

	for (unsigned i = 0; i < numEdges*2; i+=2)
		edges.push_back(Edge(i, i+1));

	return edges;
}

Edges createEdgesForBox(unsigned base = 0)
{
	Edges edges;

	Color color = Color(rand() % 256, rand() % 256, rand() % 256);
	edges.push_back(Edge(base+0, base+1, color));
	edges.push_back(Edge(base+2, base+3, color));
	edges.push_back(Edge(base+4, base+5, color));
	edges.push_back(Edge(base+6, base+7, color));

	edges.push_back(Edge(base+0, base+4, color));
	edges.push_back(Edge(base+1, base+5, color));
	edges.push_back(Edge(base+2, base+6, color));
	edges.push_back(Edge(base+3, base+7, color));

	edges.push_back(Edge(base+0, base+2, color));
	edges.push_back(Edge(base+1, base+3, color));
	edges.push_back(Edge(base+4, base+6, color));
	edges.push_back(Edge(base+5, base+7, color));

	return edges;
}

std::vector<std::string> split(std::string s)
{
	std::vector<std::string> l;

	std::istringstream iss(s);
	while (iss)
	{
		std::string subs;
		iss >> subs;
		l.push_back(subs);
	}

	return l;
}

Edges loadPLYEdges(std::string fileName)
{
	Edges edges;

	std::ifstream file;
	file.open(fileName.c_str());

	unsigned numVertices = 0;
	unsigned cntVertices = 0;
	unsigned numEdges = 0;
	unsigned cntEdges = 0;

	bool readVertices = false;
	bool readEdges = false;

	std::string line;
	while (std::getline(file, line))
	{
		if (line.find("vertex ") != std::string::npos)
			numVertices = atoi(line.find("vertex ") + line.substr(std::string("vertex ").size()).c_str());
		else if (line.find("edge ") != std::string::npos)
			numEdges = atoi(line.substr(line.find("edge ") + std::string("edge ").size()).c_str());
		else if (line.find("end_header") != std::string::npos)
			readVertices = true;

		else if (readVertices)
		{
			cntVertices++;
			if (cntVertices == numVertices)
			{
				readVertices = false;
				readEdges = true;
			}
		}

		else if (readEdges)
		{
			std::vector<std::string> list = split(line);
			unsigned first = atoi(list[0].c_str());
			unsigned second = atoi(list[1].c_str());
			edges.push_back(Edge(first, second));
			cntEdges++;
			if (cntEdges == numEdges)
				readEdges = false;
		}
	}

	return edges;
}

void savePLYEdges(std::string fileName, Cloud::Ptr cloudInput, Edges edgesInput)
{
	std::ofstream file;
  	file.open(fileName.c_str());

	file << "ply\n";
	file << "format ascii 1.0\n";
	file << "element vertex " << cloudInput->points.size() << "\n";
	file << "property float x\n";
	file << "property float y\n";
	file << "property float z\n";
	file << "element edge " << edgesInput.size() << "\n";
	file << "property int vertex1\n";
	file << "property int vertex2\n";
	file << "property uchar red\n";
	file << "property uchar green\n";
	file << "property uchar blue\n";
	file << "end_header\n";

	for (unsigned n = 0; n < cloudInput->points.size(); ++n)
	{
		file << cloudInput->points[n].x << " ";
		file << cloudInput->points[n].y << " ";
		file << cloudInput->points[n].z << "\n";
	}

	for (unsigned n = 0; n < edgesInput.size(); ++n)
	{
		file << edgesInput[n].start << " " << edgesInput[n].end << " " << edgesInput[n].color.red << " " << edgesInput[n].color.green << " " << edgesInput[n].color.blue << "\n";
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
	boxOut[4] = Plane(faces[minIdx][0], faces[minIdx][1], faces[minIdx][2]).cut(Line(boxTgt[0], boxTgt[4]));
	boxOut[5] = Plane(faces[minIdx][0], faces[minIdx][1], faces[minIdx][2]).cut(Line(boxTgt[1], boxTgt[5]));
	boxOut[6] = Plane(faces[minIdx][0], faces[minIdx][1], faces[minIdx][2]).cut(Line(boxTgt[2], boxTgt[6]));
	boxOut[7] = Plane(faces[minIdx][0], faces[minIdx][1], faces[minIdx][2]).cut(Line(boxTgt[3], boxTgt[7]));

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


Cloud::Ptr getBoundingBox(Cloud::Ptr cloudInput)
{
	Cloud::Ptr cloudOutput(new Cloud);

	Point min;
	Point max;

	pcl::getMinMax3D(*cloudInput, min, max);

	return toCloud(createConvexHull(min, max));
}

#include <algorithm>

bool comparePointOnX(Point p1, Point p2)
{ return p1.x < p2.x; }
bool comparePointOnY(Point p1, Point p2)
{ return p1.y < p2.y; }
bool comparePointOnZ(Point p1, Point p2)
{ return p1.z < p2.z; }

Cloud::Ptr getMinRectangle(Cloud::Ptr cloud)
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

	std::cout << pca.getEigenVectors() << std::endl;
	std::cout << pca.getEigenVectors()(0,0) << " " << pca.getEigenVectors()(0,1) << " " << pca.getEigenVectors()(0,2) << std::endl;
	std::cout << pca.getEigenVectors()(1,0) << " " << pca.getEigenVectors()(1,1) << " " << pca.getEigenVectors()(1,2) << std::endl;
	std::cout << pca.getEigenVectors()(2,0) << " " << pca.getEigenVectors()(2,1) << " " << pca.getEigenVectors()(2,2) << std::endl;

	std::sort(pointList.begin()+0, pointList.begin()+8, comparePointOnZ);
	std::sort(pointList.begin()+0, pointList.begin()+4, comparePointOnY);
	std::sort(pointList.begin()+4, pointList.begin()+8, comparePointOnY);
	std::sort(pointList.begin()+0, pointList.begin()+2, comparePointOnX);
	std::sort(pointList.begin()+2, pointList.begin()+4, comparePointOnX);
	std::sort(pointList.begin()+4, pointList.begin()+6, comparePointOnX);
	std::sort(pointList.begin()+6, pointList.begin()+8, comparePointOnX);
	Point mean = fromVec(pca.getMean());
	pointList.push_back(Point(mean.x + pca.getEigenVectors()(0) * 2, mean.y + pca.getEigenVectors()(1) * 2, mean.z + pca.getEigenVectors()(2) * 2));
	pointList.push_back(Point(mean.x + pca.getEigenVectors()(3) * 2, mean.y + pca.getEigenVectors()(4) * 2, mean.z + pca.getEigenVectors()(5) * 2));
	pointList.push_back(Point(mean.x + pca.getEigenVectors()(6) * 2, mean.y + pca.getEigenVectors()(7) * 2, mean.z + pca.getEigenVectors()(8) * 2));

	return toCloud(pointList);
}


void merge(Cloud::Ptr cloudOutput, Edges& edgesOutput, Cloud::Ptr cloudInput, Edges& edgesInput)
{
	unsigned offset = cloudOutput->points.size();
	*cloudOutput += *cloudInput;

	for (unsigned i = 0; i < edgesInput.size(); ++i)
	{
		Edge edge(edgesInput[i].start+offset, edgesInput[i].end+offset, edgesInput[i].color);
		edgesOutput.push_back(edge);
	}
}

pcl::PolygonMesh::Ptr merge(pcl::PolygonMesh::Ptr meshOutput, pcl::PolygonMesh::Ptr meshInput)
{
	Cloud cloudOutput;
	Cloud cloudInput;

	pcl::fromPCLPointCloud2(meshOutput->cloud, cloudOutput);
	pcl::fromPCLPointCloud2(meshInput->cloud, cloudInput);

	cloudOutput += cloudInput;
	unsigned offset = cloudOutput.points.size();

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
	tool.setAlpha(alpha);
	tool.reconstruct(*cloudOutput);

	return cloudOutput;
}

Cloud::Ptr convexHull(Cloud::Ptr cloudInput)
{
	Cloud::Ptr cloudOutput(new Cloud);

	pcl::ConvexHull<pcl::PointXYZ> tool;
	tool.setInputCloud(cloudInput);
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

Cloud::Ptr filterByDirection(Cloud::Ptr cloudInput, Eigen::Vector3f direction, float threshold)
{
	Cloud::Ptr cloudOutput(new Cloud);

	for (unsigned i = 0; i < cloudInput->points.size(); i+=2)
	{
		Eigen::Vector3f p1 = toVec(cloudInput->points[i]);
		Eigen::Vector3f p2 = toVec(cloudInput->points[i+1]);
		
		Eigen::Vector3f v = p1-p2;
		Eigen::Vector3f d = v/::sqrt(v.dot(v)) - direction;
		//std::cout << "d:" << d << "\n";

		
		if (d.dot(d) < threshold)
		{
			cloudOutput->points.push_back(cloudInput->points[i]);
			cloudOutput->points.push_back(cloudInput->points[i+1]);
		}
	}

	return cloudOutput;
}

unsigned cntEqualPoints(Cloud::Ptr cloud1, Cloud::Ptr cloud2)
{
	unsigned num = 0;

	for (unsigned i = 0; i < cloud1->points.size(); i++)
		for (unsigned j = 0; j < cloud2->points.size(); j++)
			{
				if ((cloud1->points[i].x == cloud2->points[j].x) and (cloud1->points[i].y == cloud2->points[j].y) and (cloud1->points[i].z == cloud2->points[j].z))
				{
					++num;
					break;
				}
			}

	return num;
} 

Point mean(Cloud::Ptr cloud)
{
	Point point;

  for (unsigned i = 0; i < cloud->points.size(); i++)
	{
		point.x += cloud->points[i].x;
		point.y += cloud->points[i].y;
		point.z += cloud->points[i].z;
	}

	point.x /= cloud->points.size();
	point.y /= cloud->points.size();
	point.z /= cloud->points.size();

	return point;
}

float minDistance(Cloud::Ptr cloud, Point point)
{
	float min = FLT_MAX;

  for (unsigned i = 0; i < cloud->points.size(); i++)
	{
		if (distance(cloud->points[i], point) < min)
			min = distance(cloud->points[i], point);
	}

	return min;
}

Eigen::Vector3f translate(Point p1, Point p2, Eigen::Vector3f n, float s, Point min, Point max)
{
		unsigned k = 1;
	
		Point p3 = p1;
		Point p4 = p2;
		while (isWithin(p3, min, max) and isWithin(p4, min, max))
		{
			p3 = addxy(p1, n*s*k);
			p4 = addxy(p2, n*s*k);
			k += 1;
		}

	return n*s*k;
}

typedef std::vector<std::pair<unsigned, unsigned> > Network;

Network extractNetwork(std::vector<Cluster>& clusters, float thresholdVolume)
{
	Network network;

		for (unsigned base = 0; base < clusters.size(); ++base)
		{
			for (unsigned ref = base+1; ref < clusters.size(); ++ref)
			{
				// look only at adjacent slices
				if (clusters[base].sliceId != clusters[ref].sliceId-1)
					continue;

				if (clusters[base].overlapsWith(clusters[ref]))
					std::cout << "overlap base:" << base << " ref:" << ref << " volume diff:" << ::fabs(clusters[base].volume() - clusters[ref].volume()) << std::endl;

				// if the volume is similar then an cluster are connected then there is a connection between the cluster
				if (::fabs(clusters[base].volume() - clusters[ref].volume()) < thresholdVolume and
					clusters[base].overlapsWith(clusters[ref]))
				{
					std::cerr << "add connection: " << base << " " << ref << std::endl;
					network.push_back(std::pair<unsigned, unsigned>(base, ref));
				}
			}
		}
 
	return network;
}

std::vector<Bar> extractBars(const Network& network)
{
	std::vector<Bar> bars;

		for (unsigned i = 0; i < network.size(); ++i)
		{
			bool added = false;
			for (unsigned j = 0; j < bars.size(); ++j)
			{
				if (network[i].first == bars[j].listCluster[bars[j].listCluster.size()-1])
				{
					// add cluster to bar
					bars[j].listCluster.push_back(network[i].second);
					added = true;
					break;
				}
			}

			if (not added)
			{
				// create new bar
				Bar bar;
				bar.listCluster.push_back(network[i].first);
				bar.listCluster.push_back(network[i].second);
				bars.push_back(bar);
			}
		}

		// filter out bars with a minimal number of components
		std::vector<Bar> bars_filtered;
		for (unsigned b = 0; b < bars.size(); ++b)
		{
			if (bars[b].listCluster.size() < 2)
				continue;

			bars_filtered.push_back(bars[b]);
		}

#if 1	
		// print filtered bars
		for (unsigned b = 0; b < bars_filtered.size(); ++b)
		{
			std::cerr << "bar:" <<  b << ":";
			for (unsigned s = 0; s < bars_filtered[b].listCluster.size(); ++s)
			{
				std::cerr << " " << bars_filtered[b].listCluster[s];
			}
			std::cerr << std::endl;
		}
#endif

	return bars_filtered;
}

Cloud::Ptr extractSides(Cloud::Ptr cloudInput, int numIterations, float minDist, float threshold, int minNumInliers, int maxNumOutliers, float maxCos)
{
		std::vector<Plane> planeList;
		std::vector<pcl::PointCloud<Point>::Ptr> inliersList;
		for (unsigned i = 0; i < numIterations and planeList.size() < 2; ++i)
		{
			Point p1 = cloudInput->points[rand() % cloudInput->points.size()];
			Point p2 = cloudInput->points[rand() % cloudInput->points.size()];
			Point p3 = cloudInput->points[rand() % cloudInput->points.size()];
			Plane plane(p1, p2, p3);
			pcl::PointCloud<Point>::Ptr inliers(new Cloud);

			if (distance(p1, p2) < minDist or distance(p1, p3) < minDist or distance(p2, p3) < minDist)	
			{
				continue;
			}

			unsigned numInliersLeft = 0;
			unsigned numInliersRight = 0;
			unsigned numOutliersLeft = 0;
			unsigned numOutliersRight = 0;
			for (unsigned j = 0; j < cloudInput->points.size(); ++j)
			{
				float dist = distPointToPlane(cloudInput->points[j], plane);
				if (fabs(dist) < threshold)
				{
					if (dist < 0)
						++numInliersLeft;
					else
						++numInliersRight;
					inliers->points.push_back(cloudInput->points[j]);
				}
				else
				{
					if (dist < 0)
						++numOutliersLeft;
					else
						++numOutliersRight;
				}
			}

			if (numInliersLeft + numInliersRight < minNumInliers)
				continue;
			if (numOutliersLeft > maxNumOutliers and numOutliersRight > maxNumOutliers)
				continue;
			if (minDistance(inliers, mean(inliers)) > 0.01)
				continue;

			bool do_add = true;
			for (unsigned j = 0; j < planeList.size(); ++j)
			{
				if (planeList[j].n.dot(plane.n) > maxCos)
				{
					//printf("similar angle\n");

					if (cntEqualPoints(inliersList[j], inliers) > 0)
					{
						//printf("identical\n");
						do_add = false;
					}
					else
					{
						//printf("parallel\n");
						do_add = false;
					}
				}
				else if (planeList[j].n.dot(plane.n) < 0.02)
				{
					//printf("right angle\n");
				}
				else
				{
					//printf("odd angle\n");
					do_add = false;
					break;
				}
			}

			if (do_add)
			{
				planeList.push_back(plane);
				inliersList.push_back(inliers);
			}

			//std::cout << plane.n(0) << " " << plane.n(1) << " " << plane.n(2) << " " << plane.d << " : "<< numInliersLeft << " " << numInliersRight << " " << numOutliersLeft << " " << numOutliersRight << std::endl;
		}

		Cloud::Ptr cloudOutput(new Cloud);

		if (planeList.size() < 2)
		{
			printf("extract 2 sides failed\n");
			return cloudOutput;
		}

		Point min, max;
		pcl::getMinMax3D(*cloudInput, min, max);
		Plane top(min, Point(min.x, max.y, min.z), Point(max.x, min.y, min.z));
		Plane bot(max, Point(min.x, max.y, max.z), Point(max.x, min.y, max.z));

		Line line = planeList[0].cut(planeList[1]);
		Point p1 = top.cut(line);
		Point p2 = bot.cut(line);

		Eigen::Vector3f k1 = getMax(translate(p1, p2, planeList[0].cut(top).d, 0.01, min, max), translate(p1, p2, planeList[0].cut(top).d, -0.01, min, max));
		Point p3 = addxy(p1, k1);
		Point p4 = addxy(p2, k1);

		Eigen::Vector3f k2 = getMax(translate(p1, p2, planeList[1].cut(top).d, 0.01, min, max), translate(p1, p2, planeList[1].cut(top).d, -0.01, min, max));
		Point p5 = addxy(p1, k2);
		Point p6 = addxy(p2, k2);

		Point p7 = addxy(p3, k2);
		Point p8 = addxy(p4, k2);
		//p7 = addxy(p5, k1);
		//p8 = addxy(p6, k1);

		//printf("p1:%f %f %f\n", p1.x, p1.y, p1.z);
		//printf("p2:%f %f %f\n", p2.x, p2.y, p2.z);
		//printf("p3:%f %f %f\n", p3.x, p3.y, p3.z);
		//printf("p4:%f %f %f\n", p4.x, p4.y, p4.z);
		//printf("p7:%f %f %f\n", p7.x, p7.y, p7.z);
		//printf("p8:%f %f %f\n", p8.x, p8.y, p8.z);

		cloudOutput->points.push_back(p1);
		cloudOutput->points.push_back(p2);
		cloudOutput->points.push_back(p3);
		cloudOutput->points.push_back(p4);
		cloudOutput->points.push_back(p5);
		cloudOutput->points.push_back(p6);
		cloudOutput->points.push_back(p7);
		cloudOutput->points.push_back(p8);

		printf("extract 2 sides succeeded\n");
		return cloudOutput;
}


//----------------------------------------------------------------

std::vector<Eigen::Vector3f> fitRect(Cloud::Ptr cloud, std::vector<Eigen::Vector3f> rect, int maxIt, int maxUnchangedIt, float initVariance)
{
	float minDist = FLT_MAX;

	float variance = initVariance;
	unsigned i = 0;
	while (i < maxIt)
	{
		i += 1;

		unsigned numUnchangedIt = 0;
		while (numUnchangedIt < maxUnchangedIt)
		{
#if 0
			unsigned ip = rand() % 4;
			Eigen::Vector3f v = rect[ip];
			rect[ip][0] += (rand()%2)*variance - variance/2;
			rect[ip][1] += (rand()%2)*variance - variance/2;
#endif

			static unsigned ix = 0;
			if (ix == 0)
			{
							rect[0][1] += variance;
							rect[1][1] += variance;
			}
			else if (ix == 1)
			{
							rect[1][0] -= variance;
							rect[2][0] -= variance;
			}
			else if (ix == 2)
			{
							rect[2][1] -= variance;
							rect[3][1] -= variance;
			}
			else if (ix == 3)
			{
							rect[3][0] += variance;
							rect[0][0] += variance;
			}

			float d = 0;
			for (unsigned j = 0; j < cloud->points.size(); ++j)
				 d += getMinDistToRect(cloud->points[j], rect);

			if (d < minDist)
			{
				minDist = d;
				numUnchangedIt = 0;
			}
			else
			{
				// undo changes
			if (ix == 0)
			{
							rect[0][1] -= variance;
							rect[1][1] -= variance;
			}
			else if (ix == 1)
			{
							rect[1][0] += variance;
							rect[2][0] += variance;
			}
			else if (ix == 2)
			{
							rect[2][1] += variance;
							rect[3][1] += variance;
			}
			else if (ix == 3)
			{
							rect[3][0] -= variance;
							rect[0][0] -= variance;
			}
				numUnchangedIt += 1;
			}

			ix = (ix+1) % 4;

			std::cout << "i:" << i << " numUnchangedIt:" << numUnchangedIt << " minDist:" << minDist << "\n";
		}

		variance = variance/2;
	}

	return rect;
}

//----------------------------------------------------------------

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

	if (toolname == "get-min-max")
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

#if 0
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
#endif

	else if (toolname == "get-bounding-box")
	{
		const char* withEdges = argv[argn++];
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		Cloud::Ptr cloudOutput = getBoundingBox(cloudInput);

		std::cerr << "Output Cloud: " << outFilename << " Size: " << cloudOutput->points.size() << std::endl;
		if (std::string(withEdges) == "withEdges")
			savePLYEdges(outFilename, cloudOutput, createEdgesForBox());
		else
			pcl::io::savePLYFile(outFilename, *cloudOutput);
	}

	else if (toolname == "get-min-rectangle")
	{
		const char* withEdges = argv[argn++];
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		Cloud::Ptr cloudOutput = getMinRectangle(cloudInput);

		std::cerr << "Output Cloud: " << outFilename << " Size: " << cloudOutput->points.size() << std::endl;
		if (std::string(withEdges) == "withEdges")
			savePLYEdges(outFilename, cloudOutput, createEdgesForBox());
		else
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

	else if (toolname == "convex-hull")
	{
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		Cloud::Ptr cloudOutput = convexHull(cloudInput);

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

	else if (toolname == "filter-by-direction")
	{
		Eigen::Vector3f direction;
		direction(0) = atof(argv[argn++]);
		direction(1) = atof(argv[argn++]);
		direction(2) = atof(argv[argn++]);
		float threshold = atof(argv[argn++]);
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		Cloud::Ptr cloudOutput = filterByDirection(cloudInput, direction, threshold);

		std::cerr << "Output Cloud Size: " << cloudOutput->points.size() << std::endl;
		pcl::io::savePLYFile(outFilename, *cloudOutput);
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
			pcl::io::savePLYFile(createFilename(outFilename, i), *cloudOutput);
		}
	}

	else if (toolname == "merge-cloud")
	{
		const char* withEdges = argv[argn++];
		std::vector<std::string> inFilenameList;
		while (argn < (argc-1))
			inFilenameList.push_back(argv[argn++]);
		const char* outFilename = argv[argn++];

		Cloud::Ptr cloudOutput(new Cloud);
		Edges edgesOutput;
		for (unsigned i = 0; i < inFilenameList.size(); ++i)
		{
			pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

			pcl::io::loadPLYFile(inFilenameList[i], *cloudInput);
			std::cerr << "Index: " << i << " Input Cloud Size: " << cloudInput->points.size() << std::endl;
			Edges edgesInput = loadPLYEdges(inFilenameList[i]);

			merge(cloudOutput, edgesOutput, cloudInput, edgesInput);
		}

		std::cerr << "Output Cloud Size: " << cloudOutput->points.size() << std::endl;
		if (std::string(withEdges) == "withEdges")
			savePLYEdges(outFilename, cloudOutput, edgesOutput);
		else
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

		Cloud::Ptr cloudOutput = cloudInput;
		Edges edgesOutput = createEdges(cloudInput->points.size()/2);

		std::cerr << "Output Cloud Size: " << cloudOutput->points.size() << std::endl;
		savePLYEdges(outFilename, cloudOutput, edgesOutput);
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

	else if (toolname == "extract-bars")
	{
		const char* fieldName = argv[argn++];
		float start = atof(argv[argn++]);
		float end = atof(argv[argn++]);
		int numSlices = atoi(argv[argn++]);
		float clusterTolerance = atof(argv[argn++]);
		int minClusterSize = atoi(argv[argn++]);
		int maxClusterSize = atoi(argv[argn++]);
		float thresholdVolume = atof(argv[argn++]);
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		std::vector<Cluster> clusters;

		for (unsigned i = 0; i < numSlices; ++i)
		{
			float min = start + (end-start)/numSlices * (i+0);
			float max = start + (end-start)/numSlices * (i+1);
			Cloud::Ptr slice = passThrough(cloudInput, fieldName, min, max);
			std::cout << "slice:" << i << std::endl;

			std::vector<Cloud::Ptr> listCluster = euclidianClustering(slice, clusterTolerance, minClusterSize, maxClusterSize);

			for (unsigned c = 0; c < listCluster.size(); ++c)
			{
				Cluster cluster;
				cluster.sliceId = i;
				cluster.cloud = listCluster[c];
				cluster.minmax = getMinMax(listCluster[c]);
				clusters.push_back(cluster);
				std::cout << "cluster:" << clusters.size()-1 << " points:" << listCluster[c]->points.size() << std::endl;
			}
		}

		Network network = extractNetwork(clusters, thresholdVolume);
		std::vector<Bar> bars = extractBars(network);

		pcl::PointCloud<Point>::Ptr cloudOutput(new Cloud);
		Edges edgesOutput;

		for (unsigned b = 0; b < bars.size(); ++b)
		{
			for (unsigned s = 0; s < bars[b].listCluster.size(); ++s)
			{
				Cloud::Ptr cloudBox = getBoundingBox(clusters[bars[b].listCluster[s]].cloud);
				Edges edgesBox = createEdgesForBox();
				merge(cloudOutput, edgesOutput, cloudBox, edgesBox);
			}
		}

		std::cerr << "Output Cloud with Edges: " << outFilename << " Size: " << cloudOutput->points.size() << " Edges: " << edgesOutput.size() << std::endl;
		savePLYEdges(outFilename, cloudOutput, edgesOutput);
	}

	else if (toolname == "extract-sides")
	{
		int numIterations = atoi(argv[argn++]);
		float minDist = atof(argv[argn++]);
		float threshold = atof(argv[argn++]);
		int minNumInliers = atoi(argv[argn++]);
		int maxNumOutliers = atoi(argv[argn++]);
		float maxCos = atof(argv[argn++]);
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		Cloud::Ptr cloudOutput = extractSides(cloudInput, numIterations, minDist, threshold, minNumInliers, maxNumOutliers, maxCos);

		savePLYEdges(outFilename, cloudOutput, cloudOutput->points.size() ? createEdgesForBox() : Edges());
	}

	else if (toolname == "align-sides")
	{
		float radian = atof(argv[argn++]);
		const char* refFilename = argv[argn++];
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

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

	else if (toolname == "reconstruct-bars")
	{
		const char* fieldName = argv[argn++];
		float start = atof(argv[argn++]);
		float end = atof(argv[argn++]);
		int numSlices = atoi(argv[argn++]);
		float clusterTolerance = atof(argv[argn++]);
		int minClusterSize = atoi(argv[argn++]);
		int maxClusterSize = atoi(argv[argn++]);
		float thresholdVolume = atof(argv[argn++]);
		int numRansacIterations = atoi(argv[argn++]);
		float minNumInliersFactor = atof(argv[argn++]);
		float maxNumOutliersFactor = atof(argv[argn++]);
		const char* inFilename = argv[argn++];
		const char* outFilename = argv[argn++];

		pcl::PointCloud<Point>::Ptr cloudInput(new Cloud);

		pcl::io::loadPLYFile(inFilename, *cloudInput);
		std::cerr << "Input Cloud Size: " << cloudInput->points.size() << std::endl;

		std::vector<Cluster> clusters;

		for (unsigned i = 0; i < numSlices; ++i)
		{
			float min = start + (end-start)/numSlices * (i+0);
			float max = start + (end-start)/numSlices * (i+1);
			Cloud::Ptr slice = passThrough(cloudInput, fieldName, min, max);
			std::cout << "slice:" << i << std::endl;

			std::vector<Cloud::Ptr> listCluster = euclidianClustering(slice, clusterTolerance, minClusterSize, maxClusterSize);

for (unsigned c = 0; c < listCluster.size(); ++c)
			{
				Cluster cluster;
				cluster.sliceId = i;
				cluster.cloud = listCluster[c];
				cluster.minmax = getMinMax(listCluster[c]);
				clusters.push_back(cluster);
				std::cout << "cluster:" << clusters.size()-1 << " points:" << listCluster[c]->points.size() << std::endl;
			}
		}

		Network network = extractNetwork(clusters, thresholdVolume);
		std::vector<Bar> bars = extractBars(network);

		unsigned num = 0;
		for (unsigned b = 0; b < bars.size(); ++b)
			for (unsigned s = 0; s < bars[b].listCluster.size(); ++s)
			{
				printf("reconstruct extract sides of cluster at:%d\n", bars[b].listCluster[s]);

				Cluster& cluster = clusters[bars[b].listCluster[s]];
				Cloud::Ptr cloud = cluster.cloud;
				Point min = cluster.minmax->points[0];
				Point max = cluster.minmax->points[1];
				cluster.box = extractSides(cloud, numRansacIterations, (max.x-min.x)/3, (max.x-min.x)/20, cloud->points.size()*minNumInliersFactor, cloud->points.size()*maxNumOutliersFactor, 0.98);

				if (not cluster.box->points.size())
					continue;

				// sanity check
				printf("volume quota: %f\n", cluster.volumeBox() / cluster.volume());
				if (cluster.volumeBox() <	cluster.volume()/2)
				{
					cluster.box = Cloud::Ptr(new Cloud);
					printf("extacted sides are invalid\n");
					continue;
				}

				num++;
			}
		printf("reconstructed %d of %d clusters\n", num, clusters.size());

		for (unsigned b = 0; b < bars.size(); ++b)
		{
			for (unsigned s = 0; s < bars[b].listCluster.size(); ++s)
			{
				Cluster& cluster = clusters[bars[b].listCluster[s]];
				Cluster& clusterPrev = clusters[bars[b].listCluster[s > 0 ? s-1 : s]];
				if (cluster.box->points.size() == 0 and clusterPrev.box->points.size() > 0)
				{
					cluster.box = clusterPrev.moveBox(cluster);
					num++;
					printf("cloned cluster %d to cluster %d\n", bars[b].listCluster[s-1], bars[b].listCluster[s]);
				}
			}

			for (int s = bars[b].listCluster.size()-1; s >= 0; --s)
			{
				Cluster& cluster = clusters[bars[b].listCluster[s]];
				Cluster& clusterPrev = clusters[bars[b].listCluster[s < bars[b].listCluster.size()-1 ? s+1 : s]];
				if (cluster.box->points.size() == 0 and clusterPrev.box->points.size() > 0)
				{
					cluster.box = clusterPrev.moveBox(cluster);
					num++;
					printf("cloned cluster %d to cluster %d\n", bars[b].listCluster[s+1], bars[b].listCluster[s]);
				}
			}
		}
		printf("reconstructed %d of %d clusters\n", num, clusters.size());

		for (unsigned b = 0; b < bars.size(); ++b)
		{
			Eigen::Vector3f d = Eigen::Vector3f::Zero();

			for (unsigned s = 1; s < bars[b].listCluster.size(); ++s)
			{
				Cluster& cluster = clusters[bars[b].listCluster[s]];
				Cluster& clusterPrev = clusters[bars[b].listCluster[s-1]];

				d += toVec(cluster.minmax->points[0]) - toVec(clusterPrev.minmax->points[0]);
				d += toVec(cluster.minmax->points[1]) - toVec(clusterPrev.minmax->points[1]);
			}

			d /= (bars[b].listCluster.size()-1)*2;
			printf("yyy:%f %f %f \n", d(0), d(1), d(2));

			for (unsigned s = 0; s < bars[b].listCluster.size(); ++s)
			{
				Cluster& cluster = clusters[bars[b].listCluster[s]];

				if (cluster.box->points.size() == 0)
				{
					cluster.approximateSides(-d);
					printf("approximated cluster %d\n", bars[b].listCluster[s]);
					num++;
				}
			}
		}
		printf("reconstructed %d of %d clusters\n", num, clusters.size());

		for (unsigned b = 0; b < bars.size(); ++b)
		{
			Line line;
			for (unsigned s = 0; s < bars[b].listCluster.size(); ++s)
			{
				Cloud::Ptr box = clusters[bars[b].listCluster[s]].box;
				if (not box->points.size())
					continue;

				line.p += toVec(box->points[0]) + toVec(box->points[4]);
				line.p += toVec(box->points[1]) + toVec(box->points[5]);
				line.p += toVec(box->points[2]) + toVec(box->points[6]);
				line.p += toVec(box->points[3]) + toVec(box->points[7]);

				line.d += toVec(box->points[0]) - toVec(box->points[4]);
				line.d += toVec(box->points[1]) - toVec(box->points[5]);
				line.d += toVec(box->points[2]) - toVec(box->points[6]);
				line.d += toVec(box->points[3]) - toVec(box->points[7]);
			}

			line.p /= (bars[b].listCluster.size()*8);
			line.d /= length(line.d);
	
			bars[b].lineCenter = line;
			printf("p:%f,%f,%f d:%f %f %f\n", line.p(0), line.p(1), line.p(2),  line.d(0), line.d(1), line.d(2));
		}

		for (unsigned b1 = 0; b1 < bars.size(); ++b1)
			for (unsigned b2 = b1+1; b2 < bars.size(); ++b2)
			{
				Line line1 = bars[b1].lineCenter;
				Line line2 = bars[b2].lineCenter;

				float dist = line1.distance(fromVec(line2.p));
				printf("b1:%d b2:%d distance:%f -> %s\n", b1, b2, dist, dist < 0.1 ? "same": "different");
			}

		for (unsigned b = 0; b < bars.size(); ++b)
			for (unsigned s = 0; s < bars[b].listCluster.size(); ++s)
			{
				Cluster& cluster = clusters[bars[b].listCluster[s]];

				if (not cluster.box->points.size())
					continue;

				printf("b:%d s:%d -> %f\n", b, s, cluster.getMeanDist());
			}


		pcl::PointCloud<Point>::Ptr cloudOutput(new Cloud);
		Edges edgesOutput;

		for (unsigned b = 0; b < bars.size(); ++b)
			for (unsigned s = 0; s < bars[b].listCluster.size(); ++s)
			{
				Cluster& cluster = clusters[bars[b].listCluster[s]];

				if (cluster.box->points.size())
				{
					Edges edgesBox = createEdgesForBox();
					merge(cloudOutput, edgesOutput, cluster.box, edgesBox);
				}
			}

		std::cerr << "Output Cloud with Edges: " << outFilename << " Size: " << cloudOutput->points.size() << " Edges: " << edgesOutput.size() << std::endl;
		savePLYEdges(outFilename, cloudOutput, edgesOutput);
	}

	else
	{
		std::cerr << "Unknown toolname:" << toolname << "\n";
	}


	return (0);
}

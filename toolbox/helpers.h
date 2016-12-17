#include <pcl/common/common.h>

//----------------------------------------------------------------
// basic type definitions and conversions

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

static Point operator + (Point p1, Point p2)
{
	return Point(p1.x+p2.x, p1.y+p2.y, p1.z+p2.z);
}

static Point operator - (Point p1, Point p2)
{
	return Point(p1.x-p2.x, p1.y-p2.y, p1.z-p2.z);
}

//----------------------------------------------------------------
// Edge type

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

	bool operator < (const Edge& o) const
	{
		return start < o.start ? true : (start == o.start ? end < o.end: false);
	}

	unsigned start;
	unsigned end;
	Color color;
};

typedef std::vector<Edge> Edges;

//----------------------------------------------------------------
// geometry functions and types

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

	float distance(Line line)
	{
		return fabs((d.cross(line.d)/length(d.cross(line.d))).dot(p-line.p));
	}

	Eigen::Vector3f p;
	Eigen::Vector3f d;
};

Eigen::Vector3f distance(Cloud::Ptr cloud1, Cloud::Ptr cloud2)
{
	Eigen::Vector3f d(Eigen::Vector3f::Zero());

	for (unsigned i = 0; i < cloud1->points.size() and i < cloud2->points.size(); ++i)
		d += toVec(cloud1->points[i]) - toVec(cloud2->points[i]);

	return d/cloud1->points.size();
}


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
		if (p.dot(n) < 0)
			n = -n;
		d = n.dot(p);
	}

	Plane(Eigen::Vector3f p_, Eigen::Vector3f n_)
	{
		p = p_;
		n = n_;
		n = n/length(n);
		if (p.dot(n) < 0)
			n = -n;
		d = n.dot(p);
	}

	Line cut(Plane plane)
	{
		Line line;

		float b = sqr(n)*sqr(plane.n)-n.dot(plane.n)*n.dot(plane.n);
		line.p = ((d*sqr(plane.n) - plane.d*(n.dot(plane.n)))*n)/b + ((plane.d*sqr(n) - d*(n.dot(plane.n)))*plane.n)/b;
		line.d = n.cross(plane.n);

		return line;
	}

	float distance(Line line)
	{
		return (-((line.p - p).dot(n))/line.d.dot(n));
	}

	Point cut(Line line)
	{
		return fromVec(Eigen::Vector3f(line.d * distance(line) + line.p));
	}

	float distance(Point point)
	{
		return toVec(point).dot(n) - d;
	}

	Eigen::Vector3f p;
	Eigen::Vector3f n;
	float d;
};

//----------------------------------------------------------------
// miscellaneous

std::string createFilename(std::string filenameBase, unsigned seq)
{
	std::stringstream ss;
	ss << filenameBase << seq << ".ply";

	return ss.str();
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


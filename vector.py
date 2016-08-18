import math

class Vector:
	def __init__(self):
		self.x = 0.0
		self.y = 0.0
		self.z = 0.0

	def __init__(self, x_, y_, z_):
		self.x = float(x_)
		self.y = float(y_)
		self.z = float(z_)

	def __add__(self, other):
		if isinstance(other, Vector):
				return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
		elif isinstance(other, (int, float)):
			return Vector(self.x + other, self.y + other, self.z + other)

	def __sub__(self, other):
		if isinstance(other, Vector):
				return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
		elif isinstance(other, (int, float)):
			return Vector(self.x - other, self.y - other, self.z - other)
	
	def __mul__(self, other):
		if isinstance(other, Vector):
			return self.x * other.x + self.y * other.y + self.z * other.z
		elif isinstance(other, (int, float)):
			return Vector(self.x * other, self.y * other, self.z * other)

	def __div__(self, other):
		if isinstance(other, Vector):
			return Vector(self.x / other.x, self.y / other.y, self.z / other.z)
		elif isinstance(other, (int, float)):
			return Vector(self.x / other, self.y / other, self.z / other)

	def cross(self, other):
		return Vector(self.y*other.z - self.z*other.y, self.z*other.x - self.x*other.z, self.x*other.y - self.y*other.x)

	def length(self):
		return math.sqrt(self.x *self.x + self.y *self.y + self.z *self.z)

def distPointLine(point, line):
	return (point - line[0]).cross(line[1]).length() / line[1].length()

def cutPlanePlane(plane1, plane2):

	n1 = vector.Vector(plane1[0], plane1[1], plane1[2])
	d1 = plane1[3]
	n2 = vector.Vector(plane2[0], plane2[1], plane2[2])
	d2 = plane2[3]

	r = n1.cross(n2)
	q = n1*((d1*(n2*n2) - d2*(n1*n2)) / ((n1*n1)*(n2*n2) - math.pow(n1*n2, 2))) + n2*((d2*(n1*n1) - d1*(n1*n2)) / ((n1*n1)*(n2*n2) - math.pow(n1*n2, 2)))

	return (r, q)

# plane in normal form
# line in parameter form
def cutPlaneLine(plane, line):
	x = (-((line[1] - plane[0])*plane[1])/(line[0]*plane[1]))
	print x
	return line[1] * x + line[0]

def convertPlaneToNormal(coord):
	p = coord[0]
	q = coord[1]-coord[0]
	r = coord[2]-coord[0]

	n = (q-p).cross(r-p)
	return (p,n)

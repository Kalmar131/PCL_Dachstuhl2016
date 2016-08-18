import copy
import math
import random
import sys
import vector

def makeLine(p1, p2):
	line = []
	line.append(p1)
	line.append(p2-p1)

	return line

def getMinDist(p, rect):
	minDist = sys.float_info.max

	l = []
	l.append(makeLine(rect[0], rect[1]))
	l.append(makeLine(rect[1], rect[2]))
	l.append(makeLine(rect[2], rect[3]))
	l.append(makeLine(rect[3], rect[0]))

	for line in l:
		d = vector.distPointLine(p, line)
		if d < minDist:
			minDist = d

	return minDist


cloud = []
numPoints = 0
readPoints = False

infile = open(sys.argv[1], 'r')
for line in iter(infile):
	if line.find('vertex') != -1:
		numPoints = int(line.split(' ')[2])
	if line.find('end_header') != -1:
		readPoints = True
		continue

	if not readPoints:
		continue

	if numPoints == 0:
		continue

	l = line.split(' ')
	cloud.append(vector.Vector(float(l[0]), float(l[1]), float(l[2])))
	numPoints -= 1

minx = sys.float_info.max
miny = 1000.0
minz = 1000.0

maxx = -sys.float_info.max
maxy = -sys.float_info.max
maxz = -sys.float_info.max

for p in cloud:
	if p.x < minx:
		minx = p.x
	if p.y < miny:
		miny = p.y
	if p.z < minz:
		minz = p.z

	if p.x > maxx:
		maxx = p.x
	if p.y > maxy:
		maxy = p.y
	if p.z > maxz:
		maxz = p.z

print minx,miny,minz
print maxx,maxy,maxz


rect = []
rect.append(vector.Vector(minx, miny, (minz+maxz)/2))
rect.append(vector.Vector(minx, maxy, (minz+maxz)/2))
rect.append(vector.Vector(maxx, maxy, (minz+maxz)/2))
rect.append(vector.Vector(maxx, miny, (minz+maxz)/2))

minDist = sys.float_info.max

variance = 0.002
i = 0
while i < 20:

	i += 1

	numUnchangedIt = 0
	while numUnchangedIt < 10:

		ip = random.randint(0,3)
		v = copy.copy(rect[ip])
		rect[ip].x += (random.randint(0,1)*variance - variance/2)
		rect[ip].y += (random.randint(0,1)*variance - variance/2)

		d = 0
		for point in cloud:
			 d += getMinDist(point, rect)

		if d < minDist:
			minDist = d
			numUnchangedIt = 0
		else:
			rect[ip] = v
			numUnchangedIt += 1

		print i, numUnchangedIt, minDist

	variance = variance/2

for p in rect:
	print p.x, p.y, p.z

#include <iostream>

#include <vector>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/organized.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/time.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/features/don.h>
#include <pcl/visualization/cloud_viewer.h>

#include <boost/filesystem.hpp>
#include <boost/lambda/bind.hpp>

using namespace std;
using namespace pcl;

// Array welches die Anzahl der Balken speichert für die
std::vector< int> arr_balken;

int filter(std::string file)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_balken (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_s (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointIndicesPtr ground (new pcl::PointIndices);

	pcl::io::loadPLYFile(file, *cloud);

	//TODO: Orginal muss noch gedownsampelt werden

	std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*

	// das morphologische Filterobjekt wird erstellt
	pcl::ProgressiveMorphologicalFilter<pcl::PointXYZ> pmf;
	pmf.setInputCloud (cloud);
	pmf.setMaxWindowSize (20);
	pmf.setSlope (0.2f);
	pmf.setInitialDistance (0.5f);
	pmf.setMaxDistance (5.0f);
	pmf.extract (ground->indices);

	// Filtern und extrahieren
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	extract.setInputCloud (cloud);
	extract.setIndices (ground);
	extract.filter (*cloud_filtered);

	// Extrahieren der nicht Bodenpunkte
	extract.setNegative (true);
	extract.filter (*cloud_filtered);

	// das Balkenwerk ohne Boden als .ply
	pcl::io::savePLYFile("utm.ply", *cloud_filtered, false);

	std::cout << "Morphologisches Filtern beendet" << std::endl; //*

	// abschneiden der Dachbalken
	pcl::PassThrough<pcl::PointXYZ> pass_balken;
	pass_balken.setInputCloud (cloud_filtered);
	pass_balken.setFilterFieldName ("z");
	pass_balken.setFilterLimits (0, 3);
	pass_balken.filter (*cloud_balken);

	std::cout << "abschneiden der Dachbalken beendet" << std::endl; //*

	//Balkenstümpfe für das Clustering
	pcl::PassThrough<pcl::PointXYZ> pass_f;
	pass_f.setInputCloud (cloud_filtered);
	pass_f.setFilterFieldName ("z");
	pass_f.setFilterLimits (1.8, 2.0);
	pass_f.filter (*cloud_s);

	// schreiben der Ergebnisse
	pcl::io::savePLYFile("balken_stuempfe.ply", *cloud_s, false);

	pcl::io::savePLYFile("balkenwerk.ply", *cloud_balken, false);

	std::cout << "Filtern beendet" << std::endl; //*

	return (0);

}

int euclidean_cluster_extraction (std::string file, double cluster_tolerance,int clustermin_size,int clustermax_size){

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_stuempfe (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPLYFile(file,*cloud_stuempfe);

	// KD Baum wird erstellt und das Euklidische Clustering durchgeführt
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (cloud_stuempfe);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance (cluster_tolerance);
	//für gedownsampelte Punktwolke, muss entsprechend angepasst werden
	ec.setMinClusterSize (clustermin_size);
	ec.setMaxClusterSize (clustermax_size);
	ec.setSearchMethod (tree);
	ec.setInputCloud (cloud_stuempfe);
	ec.extract (cluster_indices);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_centroid (new pcl::PointCloud<pcl::PointXYZ>);
	int j = 0;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
	{
		Eigen::Vector4f centroid;
		// Die Cluster werden aus den Indices in eine Punktwolke geschrieben
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
			cloud_cluster->points.push_back (cloud_stuempfe->points[*pit]); //*

		cloud_cluster->width = cloud_cluster->points.size ();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		//Für jedes Cluster wird ein Centroid gebildet, welcher den Mittelpunkt für den Min cut berechnet
		pcl::compute3DCentroid(*cloud_cluster,centroid);

		//Centroid Punkt werden in eine Cloud geschrieben
		pcl::PointXYZ central_point;
		central_point.x = centroid[0];
		central_point.y = centroid[1];
		central_point.z = centroid[2];
		cloud_centroid->points.push_back(central_point);
		/* Für Testzwecke
		std::stringstream ss;
		ss << "cloud_cluster_" << j << ".ply";
		pcl::io::savePLYFile(ss.str (), *cloud_cluster, false); //*
		 */
	   j++;
	}
	pcl::io::savePLYFile("center.ply", *cloud_centroid, false); //Für Test ob Cenbtroide auch wirklich die Mittelpunkte sind

	std::cout << "EuclideanClusterExtraction beendet" << std::endl; //*

	return (0);
}

int min_cut(std::string gesamt,std::string centroide,int min_size){

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_centroid (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_center_filtered (new pcl::PointCloud<pcl::PointXYZ>);

	//gesamtpunktwolke aus welcher die Balken segmentiert werden
	pcl::io::loadPLYFile(gesamt,*cloud_in);
	//centroid Punktwolke welche die mittelpunkte für das extrahieren liefern
	pcl::io::loadPLYFile(centroide,*cloud_centroid);

	//beseitigen der störenden seitenwände
	pcl::PassThrough<pcl::PointXYZ> pass_center;
	pass_center.setInputCloud (cloud_centroid);
	pass_center.setFilterFieldName ("y");
	pass_center.setFilterLimits (-7, 13);
	pass_center.filter (*cloud_center_filtered);

	pcl::io::savePLYFile("cloud_center_filtered.ply", *cloud_center_filtered, false);

	int x=0;

	for (pcl::PointCloud<pcl::PointXYZ>::const_iterator mx = cloud_center_filtered->points.begin();mx != cloud_center_filtered->points.end(); ++mx)
	{
		// MinCut Objekt
		pcl::MinCutSegmentation<pcl::PointXYZ> clustering;
		clustering.setInputCloud(cloud_in);
		pcl::PointCloud<pcl::PointXYZ>::Ptr foregroundPoints(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointXYZ point;
		point.x=mx->x;
		point.y=mx->y;
		point.z=-mx->z;
		std::cout << "CUT X: " << point.x << std::endl;
		std::cout << "CUT Y: " << point.y << std::endl;
		foregroundPoints->points.push_back(point);
		clustering.setForegroundPoints(foregroundPoints);
		clustering.setSigma(0.2);
		clustering.setRadius(2);
		clustering.setNumberOfNeighbours(50);
		clustering.setSourceWeight(0.8);

		std::vector <pcl::PointIndices> clusters;
		clustering.extract(clusters);

		for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_mint(new pcl::PointCloud<pcl::PointXYZ>);
			for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
				cluster_mint->points.push_back(cloud_in->points[*point]);
			cluster_mint->width = cluster_mint->points.size();
			cluster_mint->height = 1;
			cluster_mint->is_dense = true;

			int schwellenwert = cloud_in->points.size()-cluster_mint->points.size();
			//Balken muss kleiner als die Input wolke
			if(cluster_mint->points.size() > min_size && cluster_mint->points.size() < schwellenwert){
					std::stringstream sm;
					sm << "balken_downsampelt_" << x << ".ply";
					pcl::io::savePLYFile(sm.str(), *cluster_mint, false);
					std::cout << sm.str() << "extracted"<< std::endl;
					arr_balken.push_back(x);
			}
		}
		x++;
	}
	return(0);
}

int merge(std::string orginal){
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_orginal (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_balken (new pcl::PointCloud<pcl::PointXYZ>);

	// Read in the cloud data
	  pcl::PCDReader reader;
	  reader.read(orginal,*cloud_orginal);

	  std::cout << orginal<< " hat: " << cloud_orginal->points.size () << " Datenpunkte" << std::endl; //*
	  std::cout << "Es werden " << arr_balken.size () << " Balken extrahiert." << std::endl; //*

	  //TODO: i  noch via .ply files zählen beschränken
	  for (int  i=0; i<arr_balken.size();i++){
		  std::stringstream sm;
		  sm << "balken_downsampelt_" << arr_balken[i] << ".ply";
		  pcl::io::loadPLYFile(sm.str(),*cloud_balken);

		  std::cout << sm.str()<< "geladen" << std::endl; //*

		  //calculate the BB Box
		  pcl::PointXYZ minPt, maxPt,balken_pkt;
		  pcl::getMinMax3D (*cloud_balken, minPt, maxPt);

		  pcl::PointCloud<pcl::PointXYZ>::Ptr balken_merge(new pcl::PointCloud<pcl::PointXYZ>);
		  for(pcl::PointCloud<pcl::PointXYZ>::const_iterator mx = cloud_orginal->points.begin();mx != cloud_orginal->points.end(); ++mx){
			  if(mx-> x > minPt.x & mx-> x<maxPt.x){
				  if(mx-> y > minPt.y & mx-> y<maxPt.y){
					  if(mx-> z > minPt.z & mx-> z<maxPt.z){
						  	balken_pkt.x = mx->x;
						  	balken_pkt.y =mx->y;
						  	balken_pkt.z =mx->z;
						  	balken_merge->points.push_back(balken_pkt);
					  }
				  }
			  }
		  }
		std::stringstream sb;
		sb << "balken_orginal" << i << ".ply";
		pcl::io::savePLYFile(sb.str(), *balken_merge, false);
		std::cout << sb.str() << "extracted"<< std::endl;
	  }

	  return (0);
}

int main()
{
	//filter("gesamt.ply");
	euclidean_cluster_extraction("balken_stuempfe.ply",0.2,10,200);
	//min_cut(Cloud aus welcher die Balken extrahiert werden sollen, Mittelpunkte, mindestmenge eines Balkens)
	min_cut("balkenwerk.ply","center.ply",500);
	merge("gesamt.pcd");
	cout <<" ready";

	return (0);
}

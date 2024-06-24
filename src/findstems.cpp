#include "treeseg.h"
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char **argv)
{
	// Define and parse command line arguments
	po::options_description desc("Allowed options");
	desc.add_options()
	     ("nnearest", po::value<int>()->default_value(18),
	     	"set nnearest. This option sets the number of nearest neighbor "
			"points to include in the Euclidean clustering step. A higher number "
			"results in fewer, more significant clusters being detected, but may come "
			"at an increased computation cost."
			)
	     ("nmin", po::value<int>()->default_value(100),
	     	"set nmin. This option sets the minimum size of a cluster in order "
			"to be considered valid. Any clusters that end up with fewer points than "
			"this number will be discarded. It is used to eliminate very small clusters "
			"that are likely the result of noise or errors."
			)
	     ("lmin", po::value<float>()->default_value(2.5),
	     	"set lmin. This option determines the minimum length of a cylinder to be "
	        "considered valid in the 'RANSAC cylinder fits' step. Cylinders that are fitted "
			"and determined to be shorter than this length are discarded."
	     	)
	     ("stepcovmax", po::value<float>()->default_value(0.1),
	     	"set stepcovmax. stepcovmax (command line option): This option sets the "
			"maximum step-covering of the fitted cylinder. It seems to control for fitting "
			"quality, with higher values allowing a higher deviation from the cylinder model "
			"in the fitting process."
	     	)
	     ("radratiomin", po::value<float>()->default_value(0.9),
	     	"set radratiomin. This option sets the minimum radius ratio of the "
			"fitted cylinder. This value seems to control for the shape restrictions of "
			"the fitted cylinders."
	     	)
	     ("anglemax", po::value<float>()->default_value(35),
	     	"set anglemax. The maximum deviation (in degrees) from the vertical "
			"for the principal component of a cluster. It controls the angle at which stems "
			"(cylinders) can deviate from the vertical."
	     	)
	     ;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	// Extract values from command line arguments
	int nnearest = vm["nnearest"].as<int>();
	int nmin = vm["nmin"].as<int>();
	float lmin = vm["lmin"].as<float>();
	float stepcovmax = vm["stepcovmax"].as<float>();
	float radratiomin = vm["radratiomin"].as<float>();
	float anglemax = vm["anglemax"].as<float>();

	// The actual code
	std::vector<std::string> args(argv+1,argv+argc);

	pcl::PCDReader reader;
	pcl::PCDWriter writer;
	std::stringstream ss;

	std::cout << "Reading slice: " << std::flush;
	std::vector<std::string> id = getFileID(args[4]);
	pcl::PointCloud<PointTreeseg>::Ptr slice(new pcl::PointCloud<PointTreeseg>);
	reader.read(args[4],*slice);
	std::cout << "complete" << std::endl;

	std::cout << "Cluster extraction: " << std::flush;
	std::vector<pcl::PointCloud<PointTreeseg>::Ptr> clusters;
	std::vector<float> nndata = dNN(slice,nnearest);
	euclideanClustering(slice,nndata[0],nmin,clusters);
	ss.str("");
	ss << id[0] << ".intermediate.slice.clusters.pcd";
	writeClouds(clusters,ss.str(),false);
	std::cout << ss.str() << " | " << clusters.size() << std::endl;

	std::cout << "Region-based segmentation: " << std::flush;
	std::vector<pcl::PointCloud<PointTreeseg>::Ptr> regions;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	float smoothness = std::stof(args[0]);
	for(int i=0; i<clusters.size(); i++)
	{
		std::vector<pcl::PointCloud<PointTreeseg>::Ptr> tmpregions;
		estimateNormals(clusters[i],50,normals);
		regionSegmentation(clusters[i],normals,250,nmin,std::numeric_limits<int>::max(),smoothness,2,tmpregions);
		for(int j=0; j<tmpregions.size(); j++) regions.push_back(tmpregions[j]);
		normals->clear();
	}
	ss.str("");
	ss << id[0] << ".intermediate.slice.clusters.regions.pcd";
	writeClouds(regions,ss.str(),false);
	std::cout << ss.str() << " | " << regions.size() << std::endl;

	std::cout << "RANSAC cylinder fits: " << std::flush;
	std::vector<std::pair<float,pcl::PointCloud<PointTreeseg>::Ptr>> cylinders;
	// nnearest = 60;
	float dmin = std::stof(args[1]);
	float dmax = std::stof(args[2]);
	std::ifstream coordfile;
	coordfile.open(args[3]);
	float coords[4];
	int n = 0;
	while(coordfile >> coords[n])
		n++;
	coordfile.close();
	float xmin = coords[0];
	float xmax = coords[1];
	float ymin = coords[2];
	float ymax = coords[3];
	for(int i=0; i<regions.size(); i++)
	{
		cylinder cyl;
		fitCylinder(regions[i],nnearest,true,true,cyl);
		if(cyl.ismodel == true)
		{
			if(cyl.rad*2 >= dmin && cyl.rad*2 <= dmax && cyl.len >= lmin)
			{
				if(cyl.stepcov <= stepcovmax && cyl.radratio > radratiomin)
				{
					if(cyl.x >= xmin && cyl.x <= xmax)
					{
						if(cyl.y >= ymin && cyl.y <= ymax)
						{
							cylinders.push_back(std::make_pair(cyl.rad,cyl.inliers));
						}
					}
				}
			}
		}
	}
	std::sort(cylinders.rbegin(),cylinders.rend());
	std::vector<pcl::PointCloud<PointTreeseg>::Ptr> cyls;
	for(int i=0;i<cylinders.size();i++) cyls.push_back(cylinders[i].second);
	ss.str("");
	ss << id[0] << ".intermediate.slice.clusters.regions.cylinders.pcd";
	writeClouds(cyls,ss.str(),false);
	std::cout << ss.str() << " | " << cyls.size() << std::endl;

	std::cout << "Principal component trimming: " << std::flush;
	std::vector<int> idx;
	for(int j=0; j<cyls.size(); j++)
	{
		Eigen::Vector4f centroid;
		Eigen::Matrix3f covariancematrix;
		Eigen::Matrix3f eigenvectors;
		Eigen::Vector3f eigenvalues;
		computePCA(cyls[j],centroid,covariancematrix,eigenvectors,eigenvalues);
		Eigen::Vector4f gvector(eigenvectors(0,2),eigenvectors(1,2),0,0);
		Eigen::Vector4f cvector(eigenvectors(0,2),eigenvectors(1,2),eigenvectors(2,2),0);
		float angle = pcl::getAngle3D(gvector,cvector) * (180/M_PI);
		if(angle >= (90 - anglemax) && angle <= (90 + anglemax)) idx.push_back(j);
	}
	std::vector<pcl::PointCloud<PointTreeseg>::Ptr> pca;
	for(int k=0; k<idx.size(); k++) pca.push_back(cyls[idx[k]]);
	ss.str("");
	ss << id[0] << ".intermediate.slice.clusters.regions.cylinders.principal.pcd";
	writeClouds(pca,ss.str(),false);
	std::cout << ss.str() << " | " << pca.size() << std::endl;

	std::cout << "Concatenating stems: " << std::flush;
	std::vector<pcl::PointCloud<PointTreeseg>::Ptr> stems;
	stems = pca;
	catIntersectingClouds(stems);
	ss.str("");
	ss << id[0] << ".intermediate.slice.clusters.regions.cylinders.principal.cat.pcd";
	writeClouds(stems,ss.str(),false);
	for(int m=0; m<stems.size(); m++)
	{
		ss.str("");
		ss << id[0] << ".cluster." << m << ".pcd";
		writer.write(ss.str(),*stems[m],true);
	}
	std::cout << stems.size() << std::endl;
	//
	return 0;
}
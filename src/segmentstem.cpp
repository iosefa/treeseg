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
    ("nnearest", po::value<int>()->default_value(60),
    "This option specifies the number of nearest neighbours to be "
    "considered for the RANSAC cylinder fit algorithm. Larger values imply higher "
    "accuracy but slower speed.")
    ("expansionfactor", po::value<float>()->default_value(12.0),
    "This option refers to the factor by which the radius of the initially "
    "fitted cylinder gets expanded during segmentation.")
    ("groundidx", po::value<int>()->default_value(0),
    "It is used in the process of separating the ground from the rest of the point cloud. The default is the first point (index 0).")
    ("ground", po::value<float>()->default_value(0.f),
    "This value determines the threshold separating the ground from the remaining point cloud in terms of its elevation.")
    ("dthreshold", po::value<float>()->default_value(0.5f),
    "It represents a threshold used during the RANSAC process to help determine the ground plane.")
    ("nweight", po::value<float>()->default_value(0.75f),
    "This term refers to the weighting of normal vectors when fitting the ground plane.")
    ("angle", po::value<int>()->default_value(30),
    "This input specifies the maximum angle (in degrees) allowed for the normal vector of points when trying to fit the ground plane during segmentation.")
    ("nneighbours", po::value<int>()->default_value(250),
    "This parameter tells the algorithm how many neighbours should be considered for each point during region growing in the point cloud.")
    ("nmin", po::value<int>()->default_value(3),
    "This parameter sets the minimum number of points a region must have to be considered valid in region-based segmentation.")
    ("nmax", po::value<int>()->default_value(std::numeric_limits<int>::max()),
    "This value is the maximum number of points that a region can contain in region-based segmentation.")
    ("curvature", po::value<float>()->default_value(1.0f),
    "This threshold for curvature value is used during region growing in segmentation. Regions with higher curvature aren't included in the same segment.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // Extract values from command line arguments
    int nnearest = vm["nnearest"].as<int>();
    float expansionfactor = vm["expansionfactor"].as<float>();
    int groundidx = vm["groundidx"].as<int>();
    float ground = vm["ground"].as<float>();
    float dthreshold = vm["dthreshold"].as<float>();
    float nweight = vm["nweight"].as<float>();
    int angle = vm["angle"].as<int>();
    int nneighbours = vm["nneighbours"].as<int>();
    int nmin = vm["nmin"].as<int>();
    int nmax = vm["nmax"].as<int>();
    float curvature = vm["curvature"].as<float>();

    std::vector<std::string> args(argv+1,argv+argc);
    pcl::PCDReader reader;
    pcl::PCDWriter writer;
    std::stringstream ss;
    std::cout << "Reading plot-level cloud: " << std::flush;
    pcl::PointCloud<PointTreeseg>::Ptr plot(new pcl::PointCloud<PointTreeseg>);
    readTiles(args,plot);
    std::cout << "complete" << std::endl;

    for(int i=1;i<getTilesStartIdx(args);i++)
    {
        std::vector<std::string> id = getFileID(args[i]);
        pcl::PointCloud<PointTreeseg>::Ptr foundstem(new pcl::PointCloud<PointTreeseg>);
        reader.read(args[i],*foundstem);
        std::cout << "RANSAC cylinder fit: " << std::flush;
        cylinder cyl;
        fitCylinder(foundstem, nnearest, false, false, cyl);
        if(cyl.rad < 0.05) cyl.rad = 0.05;
        std::cout << cyl.rad << std::endl;
        std::cout << "Segmenting extended cylinder: " << std::flush;
        pcl::PointCloud<PointTreeseg>::Ptr volume(new pcl::PointCloud<PointTreeseg>);
        cyl.rad = cyl.rad * expansionfactor;
        spatial3DCylinderFilter(plot,cyl,volume);
        ss.str("");
        ss << id[0] << ".c." << id[1] << ".pcd";
        writer.write(ss.str(),*volume,true);
        std::cout << ss.str() << std::endl;

        std::cout << "Segmenting ground returns: " << std::flush;
        pcl::PointCloud<PointTreeseg>::Ptr bottom(new pcl::PointCloud<PointTreeseg>);
        pcl::PointCloud<PointTreeseg>::Ptr top(new pcl::PointCloud<PointTreeseg>);
        pcl::PointCloud<PointTreeseg>::Ptr vnoground(new pcl::PointCloud<PointTreeseg>);
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        std::sort(volume->points.begin(),volume->points.end(),sortCloudByZ);
        float xmin = volume->points[groundidx].x;
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*volume, min_pt, max_pt);
        spatial1DFilter(volume, "z", ground - 1, ground + 3.5, bottom);
        spatial1DFilter(volume, "z", ground + 3.5, max_pt[2], top);
        estimateNormals(bottom, 250, normals);
        fitPlane(bottom, normals, dthreshold, inliers, nweight, angle);
        extractIndices(bottom, inliers,true,vnoground);
        *vnoground += *top;
        ss.str("");
        ss << id[0] << ".c.ng." << id[1] << ".pcd";
        writer.write(ss.str(),*vnoground,true);
        std::cout << ss.str() << std::endl;

        std::cout << "Region-based segmentation: " << std::flush;
        std::vector<pcl::PointCloud<PointTreeseg>::Ptr> regions;
        normals->clear();
        estimateNormals(vnoground, 50, normals);
        float smoothness = std::stof(args[0]);
        regionSegmentation(vnoground,normals,nneighbours,nmin,nmax,smoothness,1,regions);

        ss.str("");
        ss << id[0] << ".c.ng.r." << id[1] << ".pcd";
        writeClouds(regions,ss.str(),false);
        std::cout << ss.str() << std::endl;

        std::cout << "Correcting stem: " << std::flush;
        pcl::PointCloud<PointTreeseg>::Ptr stem(new pcl::PointCloud<PointTreeseg>);
        int idx = findClosestIdx(foundstem,regions,true);
        *stem += *regions[idx];
        ss.str("");
        ss << id[0] << ".stem." << id[1] << ".pcd";
        writer.write(ss.str(),*stem,true);
        std::cout << ss.str() << std::endl;
    }
    return 0;
}

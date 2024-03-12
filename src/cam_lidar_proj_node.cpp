//
// Created by usl on 4/10/19.
//

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include "draconis_demo_custom_msgs/ImagePointcloudMsg.h"
#include "draconis_demo_custom_msgs/ImagesAndPointcloudMsg.h"

#include <tf/transform_listener.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Geometry>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/passthrough.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>

#include <pcl/filters/statistical_outlier_removal.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <iostream>
#include <fstream>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
                                                        sensor_msgs::Image> SyncPolicy;

class lidarImageProjection 
{
private:

    ros::NodeHandle nh;

    ros::Subscriber imageAndCloudSub;
    ros::Subscriber imagesAndCloudSub;

    ros::Publisher cloud_pub;
    ros::Publisher image_pub;

    ros::Duration samplingDuration;
    ros::Time prevCycleTime;

    cv::Mat c_R_l, tvec;
    cv::Mat rvec;
    std::string result_str;
    Eigen::Matrix4d C_T_L, L_T_C;
    Eigen::Matrix3d C_R_L, L_R_C;
    Eigen::Quaterniond C_R_L_quatn, L_R_C_quatn;
    Eigen::Vector3d C_t_L, L_t_C;

    bool project_only_plane;
    cv::Mat projection_matrix;
    cv::Mat distCoeff;

    std::vector<cv::Point3d> objectPoints_L, objectPoints_C;
    std::vector<cv::Point2d> imagePoints;

    sensor_msgs::PointCloud2 out_cloud_ros;

    std::string lidar_frameId;

    pcl::PointCloud<pcl::PointXYZRGB> out_cloud_pcl;
    cv::Mat image_in;

    int cloud_cutoff_distance;

    std::string cam_config_file_path;
    int image_width, image_height;

    std::string camera_name;

    double x_min, x_max;
    double y_min, y_max;
    double z_min, z_max;

    double ransac_threshold = 0.01;

    int sor_mean_k = 50;
    double sor_std_dev = 1.0;

public:
    lidarImageProjection() 
    {
        ROS_INFO("Start initialization...");

        //--- Read params
        std::string topic_input_image_and_cloud = readParam<std::string>(nh, "topic_input_image_and_cloud");
        std::string topic_input_images_and_cloud = readParam<std::string>(nh, "topic_input_images_and_cloud");
        std::string lidarOutTopic = readParam<std::string>(nh, "topic_output_velodyne_cloud");
        std::string imageOutTopic = readParam<std::string>(nh, "topic_output_projected_image");

        cloud_cutoff_distance = readParam<int>(nh, "dist_cut_off");
        camera_name = readParam<std::string>(nh, "camera_name");

        result_str = readParam<std::string>(nh, "result_file");
        project_only_plane = readParam<bool>(nh, "project_only_plane");

        cam_config_file_path = readParam<std::string>(nh, "cam_config_file_path");

        projection_matrix = cv::Mat::zeros(3, 3, CV_64F);
        distCoeff = cv::Mat::zeros(5, 1, CV_64F);
        
        readCameraParams(cam_config_file_path,
                         image_height,
                         image_width,
                         distCoeff,
                         projection_matrix);

        
        x_min = readParam<double>(nh, "x_min");
        x_max = readParam<double>(nh, "x_max");
        y_min = readParam<double>(nh, "y_min");
        y_max = readParam<double>(nh, "y_max");
        z_min = readParam<double>(nh, "z_min");
        z_max = readParam<double>(nh, "z_max");

        ransac_threshold = readParam<double>(nh, "ransac_threshold");

        sor_mean_k = readParam<double>(nh, "sor_mean_k");
        sor_std_dev = readParam<double>(nh, "sor_std_dev");


        //--- Other initialization
        samplingDuration = ros::Duration(0.2);  // in sec
        prevCycleTime = ros::Time(1);

        C_T_L = Eigen::Matrix4d::Identity();
        c_R_l = cv::Mat::zeros(3, 3, CV_64F);
        tvec = cv::Mat::zeros(3, 1, CV_64F);

        ROS_INFO("result file: %s", result_str.c_str());
        std::ifstream myReadFile(result_str.c_str());
        std::string word;
        int i = 0;
        int j = 0;

        while (myReadFile >> word)
        {
            ROS_INFO("word: %s", word.c_str());
            C_T_L(i, j) = std::stod(word);
            ROS_INFO("%.3f", C_T_L(i, j));

            j++;

            if (j>3) 
            {
                j = 0;
                i++;
            }
        }

        L_T_C = C_T_L.inverse();

        C_R_L = C_T_L.block(0, 0, 3, 3);
        C_t_L = C_T_L.block(0, 3, 3, 1);

        L_R_C = L_T_C.block(0, 0, 3, 3);
        L_t_C = L_T_C.block(0, 3, 3, 1);

        cv::eigen2cv(C_R_L, c_R_l);
        C_R_L_quatn = Eigen::Quaterniond(C_R_L);
        L_R_C_quatn = Eigen::Quaterniond(L_R_C);
        cv::Rodrigues(c_R_l, rvec);
        cv::eigen2cv(C_t_L, tvec);


        //--- Subscribers
        imageAndCloudSub = nh.subscribe(topic_input_image_and_cloud, 5, &lidarImageProjection::imageAndCloudCallback, this);
        imagesAndCloudSub = nh.subscribe(topic_input_images_and_cloud, 5, &lidarImageProjection::imagesAndCloudCallback, this);

        //--- Publishers
        cloud_pub = nh.advertise<sensor_msgs::PointCloud2>(lidarOutTopic, 1);
        image_pub = nh.advertise<sensor_msgs::Image>(imageOutTopic, 1);

        ROS_INFO("Initialization done !");

    }


    void readCameraParams(std::string cam_config_file_path,
                          int &image_height,
                          int &image_width,
                          cv::Mat &D,
                          cv::Mat &K) 
    {
        cv::FileStorage fs_cam_config(cam_config_file_path, cv::FileStorage::READ);

        if(!fs_cam_config.isOpened())
            std::cerr << "Error: Wrong path: " << cam_config_file_path << std::endl;

        fs_cam_config["image_height"] >> image_height;
        fs_cam_config["image_width"] >> image_width;
        fs_cam_config["k1"] >> D.at<double>(0);
        fs_cam_config["k2"] >> D.at<double>(1);
        fs_cam_config["p1"] >> D.at<double>(2);
        fs_cam_config["p2"] >> D.at<double>(3);
        fs_cam_config["k3"] >> D.at<double>(4);
        fs_cam_config["fx"] >> K.at<double>(0, 0);
        fs_cam_config["fy"] >> K.at<double>(1, 1);
        fs_cam_config["cx"] >> K.at<double>(0, 2);
        fs_cam_config["cy"] >> K.at<double>(1, 2);
    }


    template <typename T>
    T readParam(ros::NodeHandle &n, std::string name)
    {
        T ans;
        if (n.getParam(name, ans)){
            ROS_INFO_STREAM("Loaded " << name << ": " << ans);
        } else {
            ROS_ERROR_STREAM("Failed to load " << name);
            n.shutdown();
        }
        return ans;
    }


    void imageAndCloudCallback(const draconis_demo_custom_msgs::ImagePointcloudMsgConstPtr &msg)
    {
        ROS_INFO("<< node::imageAndCloudCallback >>");

        callback(msg);
    }


    void callback(const draconis_demo_custom_msgs::ImagePointcloudMsgConstPtr &msg) 
    {

        sensor_msgs::PointCloud2 cloud_msg = msg->pointcloud;
        sensor_msgs::Image image_msg = msg->image;

        lidar_frameId = cloud_msg.header.frame_id;

        objectPoints_L.clear();
        objectPoints_C.clear();
        imagePoints.clear();

        // publishTransforms();
        
        boost::shared_ptr<void const> tracked_object;
        image_in = cv_bridge::toCvShare(image_msg, tracked_object, "bgr8")->image;

        double fov_x, fov_y;
        fov_x = 2*atan2(image_width, 2*projection_matrix.at<double>(0, 0))*180/CV_PI;
        fov_y = 2*atan2(image_height, 2*projection_matrix.at<double>(1, 1))*180/CV_PI;

        double max_range, min_range;
        max_range = -INFINITY;
        min_range = INFINITY;

        pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        if (project_only_plane)  // only project onto planes
        {
            in_cloud = extractPlane(cloud_msg);

            for(size_t i = 0; i < in_cloud->points.size(); i++) 
            {
                objectPoints_L.push_back(cv::Point3d(in_cloud->points[i].x, in_cloud->points[i].y, in_cloud->points[i].z));
            }

            cv::projectPoints(objectPoints_L, 
                              rvec, 
                              tvec, 
                              projection_matrix, 
                              distCoeff, 
                              imagePoints, 
                              cv::noArray());

        } 
        else 
        {
            pcl::fromROSMsg(cloud_msg, *in_cloud);

            for(size_t i = 0; i < in_cloud->points.size(); i++) 
            {

                // Reject points behind the LiDAR(and also beyond certain distance)
                // if(in_cloud->points[i].x < 0 || in_cloud->points[i].x > cloud_cutoff_distance)
                //     continue;

                Eigen::Vector4d pointCloud_L;
                pointCloud_L[0] = in_cloud->points[i].x;
                pointCloud_L[1] = in_cloud->points[i].y;
                pointCloud_L[2] = in_cloud->points[i].z;
                pointCloud_L[3] = 1;

                Eigen::Vector3d pointCloud_C;
                pointCloud_C = C_T_L.block(0, 0, 3, 4)*pointCloud_L;

                double X = pointCloud_C[0];
                double Y = pointCloud_C[1];
                double Z = pointCloud_C[2];

                double Xangle = atan2(X, Z)*180/CV_PI;
                double Yangle = atan2(Y, Z)*180/CV_PI;

                if(Xangle < -fov_x/2 || Xangle > fov_x/2)
                    continue;

                if(Yangle < -fov_y/2 || Yangle > fov_y/2)
                    continue;

                double range = sqrt(X*X + Y*Y + Z*Z);

                if(range > max_range) 
                {
                    max_range = range;
                }

                if(range < min_range) 
                {
                    min_range = range;
                }

                objectPoints_L.push_back(cv::Point3d(pointCloud_L[0], pointCloud_L[1], pointCloud_L[2]));
                objectPoints_C.push_back(cv::Point3d(X, Y, Z));
            }

            cv::projectPoints(objectPoints_L, 
                              rvec, 
                              tvec, 
                              projection_matrix, 
                              distCoeff, 
                              imagePoints, 
                              cv::noArray());
        }

        /// Color the Point Cloud
        colorPointCloud();

        pcl::toROSMsg(out_cloud_pcl, out_cloud_ros);
        out_cloud_ros.header.frame_id = cloud_msg.header.frame_id;
        out_cloud_ros.header.stamp = cloud_msg.header.stamp;

        cloud_pub.publish(out_cloud_ros);

        /// Color Lidar Points on the image a/c to distance
        colorLidarPointsOnImage(min_range, max_range);

        sensor_msgs::ImagePtr out_msg =
                cv_bridge::CvImage(std_msgs::Header(), "bgr8", image_in).toImageMsg();
        image_pub.publish(out_msg);

        // cv::Mat image_resized;
        // cv::resize(lidarPtsImg, image_resized, cv::Size(), 0.25, 0.25);
        // cv::imshow("view", image_resized);
        // cv::waitKey(10);
    }
    

    void imagesAndCloudCallback(const draconis_demo_custom_msgs::ImagesAndPointcloudMsgConstPtr &msg)
    {
        ROS_INFO("<< node::imageAndCloudCallback >>");

        msgHandler(msg);
    }


    void msgHandler(const draconis_demo_custom_msgs::ImagesAndPointcloudMsgConstPtr &msg) 
    {

        sensor_msgs::PointCloud2 cloud_msg = msg->pointcloud;
        sensor_msgs::Image image_msg = msg->image_front;

        lidar_frameId = cloud_msg.header.frame_id;

        objectPoints_L.clear();
        objectPoints_C.clear();
        imagePoints.clear();

        // publishTransforms();
        
        boost::shared_ptr<void const> tracked_object;
        image_in = cv_bridge::toCvShare(image_msg, tracked_object, "bgr8")->image;

        double fov_x, fov_y;
        fov_x = 2*atan2(image_width, 2*projection_matrix.at<double>(0, 0))*180/CV_PI;
        fov_y = 2*atan2(image_height, 2*projection_matrix.at<double>(1, 1))*180/CV_PI;

        double max_range, min_range;
        max_range = -INFINITY;
        min_range = INFINITY;

        pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        if (project_only_plane)  // only project onto planes
        {
            in_cloud = extractPlane(cloud_msg);

            for(size_t i = 0; i < in_cloud->points.size(); i++) 
            {
                objectPoints_L.push_back(cv::Point3d(in_cloud->points[i].x, in_cloud->points[i].y, in_cloud->points[i].z));
            }

            cv::projectPoints(objectPoints_L, 
                              rvec, 
                              tvec, 
                              projection_matrix, 
                              distCoeff, 
                              imagePoints, 
                              cv::noArray());

        } 
        else 
        {
            pcl::fromROSMsg(cloud_msg, *in_cloud);

            for(size_t i = 0; i < in_cloud->points.size(); i++) 
            {

                // Reject points behind the LiDAR(and also beyond certain distance)
                // if(in_cloud->points[i].x < 0 || in_cloud->points[i].x > cloud_cutoff_distance)
                //     continue;

                Eigen::Vector4d pointCloud_L;
                pointCloud_L[0] = in_cloud->points[i].x;
                pointCloud_L[1] = in_cloud->points[i].y;
                pointCloud_L[2] = in_cloud->points[i].z;
                pointCloud_L[3] = 1;

                Eigen::Vector3d pointCloud_C;
                pointCloud_C = C_T_L.block(0, 0, 3, 4)*pointCloud_L;

                double X = pointCloud_C[0];
                double Y = pointCloud_C[1];
                double Z = pointCloud_C[2];

                double Xangle = atan2(X, Z)*180/CV_PI;
                double Yangle = atan2(Y, Z)*180/CV_PI;

                if(Xangle < -fov_x/2 || Xangle > fov_x/2)
                    continue;

                if(Yangle < -fov_y/2 || Yangle > fov_y/2)
                    continue;

                double range = sqrt(X*X + Y*Y + Z*Z);

                if(range > max_range) 
                {
                    max_range = range;
                }

                if(range < min_range) 
                {
                    min_range = range;
                }

                objectPoints_L.push_back(cv::Point3d(pointCloud_L[0], pointCloud_L[1], pointCloud_L[2]));
                objectPoints_C.push_back(cv::Point3d(X, Y, Z));
            }

            cv::projectPoints(objectPoints_L, 
                              rvec, 
                              tvec, 
                              projection_matrix, 
                              distCoeff, 
                              imagePoints, 
                              cv::noArray());
        }

        /// Color the Point Cloud
        colorPointCloud();

        pcl::toROSMsg(out_cloud_pcl, out_cloud_ros);
        out_cloud_ros.header.frame_id = cloud_msg.header.frame_id;
        out_cloud_ros.header.stamp = cloud_msg.header.stamp;

        cloud_pub.publish(out_cloud_ros);

        /// Color Lidar Points on the image a/c to distance
        colorLidarPointsOnImage(min_range, max_range);

        sensor_msgs::ImagePtr out_msg =
                cv_bridge::CvImage(std_msgs::Header(), "bgr8", image_in).toImageMsg();
        image_pub.publish(out_msg);

        // cv::Mat image_resized;
        // cv::resize(lidarPtsImg, image_resized, cv::Size(), 0.25, 0.25);
        // cv::imshow("view", image_resized);
        // cv::waitKey(10);
    }


    pcl::PointCloud<pcl::PointXYZ >::Ptr extractPlane(sensor_msgs::PointCloud2 &cloud_msg) 
    {
        ROS_INFO("<< node::extractPlane >>");

        //--- Convert ROS cloud to PCL cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(cloud_msg, *in_cloud);

        //--- Filter cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        filterCloud(in_cloud, cloud_filtered);
        cloud_filtered->header = in_cloud->header;

        //--- Plane Segmentation, using RANSAC algorithm
        pcl::PointCloud<pcl::PointXYZ >::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ >::Ptr plane_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(
                new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud_filtered));
        pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);

        ransac.setDistanceThreshold(ransac_threshold);
        ransac.computeModel();

        std::vector<int> inliers_indicies;
        ransac.getInliers(inliers_indicies);
        pcl::copyPointCloud<pcl::PointXYZ>(*cloud_filtered, inliers_indicies, *plane);

        //--- Statistical Outlier Removal
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(plane);
        sor.setMeanK (sor_mean_k);
        sor.setStddevMulThresh (sor_std_dev);
        sor.filter (*plane_filtered);

        return plane_filtered;
    }


    void filterCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &inputCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &outputCloud)
    {
        ROS_INFO("<< node::filterCloud >>");

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_x(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZ>);

        /// Pass through filters
        pcl::PassThrough<pcl::PointXYZ> pass_x;
        pass_x.setInputCloud(inputCloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(x_min, x_max);
        pass_x.filter(*cloud_filtered_x);

        pcl::PassThrough<pcl::PointXYZ> pass_y;
        pass_y.setInputCloud(cloud_filtered_x);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(y_min, y_max);
        pass_y.filter(*cloud_filtered_y);

        pcl::PassThrough<pcl::PointXYZ> pass_z;
        pass_z.setInputCloud(cloud_filtered_y);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min, z_max);
        pass_z.filter(*cloud_filtered_z);

        outputCloud = cloud_filtered_z;
    }



    cv::Vec3b atf(cv::Mat rgb, cv::Point2d xy_f)
    {
        cv::Vec3i color_i;
        color_i.val[0] = color_i.val[1] = color_i.val[2] = 0;

        int x = xy_f.x;
        int y = xy_f.y;

        for (int row = 0; row <= 1; row++){
            for (int col = 0; col <= 1; col++){
                if((x+col)< rgb.cols && (y+row) < rgb.rows) {
                    cv::Vec3b c = rgb.at<cv::Vec3b>(cv::Point(x + col, y + row));
                    for (int i = 0; i < 3; i++){
                        color_i.val[i] += c.val[i];
                    }
                }
            }
        }

        cv::Vec3b color;
        for (int i = 0; i < 3; i++){
            color.val[i] = color_i.val[i] / 4;
        }
        return color;
    }


    void publishTransforms() 
    {
        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;
        tf::quaternionEigenToTF(L_R_C_quatn, q);
        transform.setOrigin(tf::Vector3(L_t_C(0), L_t_C(1), L_t_C(2)));
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), lidar_frameId, camera_name));
    }


    void colorPointCloud() 
    {
        out_cloud_pcl.points.clear();
        out_cloud_pcl.resize(objectPoints_L.size());

        for(size_t i = 0; i < objectPoints_L.size(); i++) {
            cv::Vec3b rgb = atf(image_in, imagePoints[i]);
            pcl::PointXYZRGB pt_rgb(rgb.val[2], rgb.val[1], rgb.val[0]);
            pt_rgb.x = objectPoints_L[i].x;
            pt_rgb.y = objectPoints_L[i].y;
            pt_rgb.z = objectPoints_L[i].z;
            out_cloud_pcl.push_back(pt_rgb);
        }
    }


    void colorLidarPointsOnImage(double min_range, double max_range) 
    {
        for(size_t i = 0; i < imagePoints.size(); i++) {
            double X = objectPoints_C[i].x;
            double Y = objectPoints_C[i].y;
            double Z = objectPoints_C[i].z;
            double range = sqrt(X*X + Y*Y + Z*Z);
            double red_field = 255*(range - min_range)/(max_range - min_range);
            double green_field = 255*(max_range - range)/(max_range - min_range);

            cv::circle(image_in, 
                        imagePoints[i], 
                        2,
                       CV_RGB(red_field, green_field, 0), 
                       -1, 
                       1, 
                       0);
        }
    }

};


int main(int argc, char** argv) {
    ros::init(argc, argv, "cam_lidar_proj");
    lidarImageProjection lip;
    ros::spin();
    return 0;
}
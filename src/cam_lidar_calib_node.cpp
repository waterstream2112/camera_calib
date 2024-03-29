//
// Created by usl on 4/6/19.
//

#include <algorithm>
#include <random>
#include <chrono>
#include <ctime>


#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"
#include "draconis_demo_custom_msgs/ImagePointcloudMsg.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model.h>

#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <cv_bridge/cv_bridge.h>

#include <Eigen/Dense>

#include <calibration_error_term.h>

#include "ceres/ceres.h"
#include "glog/logging.h"

#include "ceres/rotation.h"
#include "ceres/covariance.h"

#include <fstream>
#include <iostream>


class camLidarCalib {

private:
    ros::NodeHandle nh;

    // message_filters::Subscriber<sensor_msgs::PointCloud2> *cloud_sub;
    // message_filters::Subscriber<sensor_msgs::Image> *image_sub;
    // message_filters::Synchronizer<SyncPolicy> *sync;

    // ros::Subscriber imageSub;
    // ros::Subscriber cloudSub;
    ros::Subscriber imageAndCloudSub;
    ros::Subscriber imagesAndCloudSub;

    ros::Publisher filtered_cloud_pub;
    ros::Publisher plane_cloud_pub;

    sensor_msgs::PointCloud2ConstPtr cloudMsgPtr;
    sensor_msgs::ImageConstPtr imageMsgPtr;

    ros::Duration samplingDuration;
    ros::Time prevCycleTime;

    cv::Mat image_in;
    cv::Mat image_resized;
    cv::Mat projection_matrix;
    cv::Mat distCoeff;
    std::vector<cv::Point2f> image_points;
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> projected_points;
    bool boardDetectedInCam;

    // Size in meter of a square on checkerboard
    double dx, dy;
    
    int checkerboard_rows, checkerboard_cols;
    int min_points_on_plane;
    int max_points_on_plane;

    cv::Mat tvec, rvec;
    cv::Mat C_R_W;
    Eigen::Matrix3d c_R_w;
    Eigen::Vector3d c_t_w;
    Eigen::Vector3d r3;
    Eigen::Vector3d r3_old;
    Eigen::Vector3d Nc;

    std::vector<Eigen::Vector3d> lidar_points;
    std::vector<std::vector<Eigen::Vector3d>> all_lidar_points;
    std::vector<Eigen::Vector3d> all_normals;

    std::string result_str, result_rpy;

    std::string camera_in_topic;
    std::string lidar_in_topic;
    std::string image_and_cloud_in_topic;
    std::string images_and_cloud_in_topic;

    int num_views;

    std::string cam_config_file_path;
    int image_width, image_height;

    double x_min, x_max;
    double y_min, y_max;
    double z_min, z_max;

    double ransac_threshold;
    int no_of_initializations;
    std::string initializations_file;
    std::ofstream init_file;

public:
    camLidarCalib(ros::NodeHandle n) {
        nh = n;
        // camera_in_topic = readParam<std::string>(nh, "camera_in_topic");
        // lidar_in_topic = readParam<std::string>(nh, "lidar_in_topic");
        image_and_cloud_in_topic = readParam<std::string>(nh, "image_and_cloud_in_topic");
        images_and_cloud_in_topic = readParam<std::string>(nh, "images_and_cloud_in_topic");

        // cloud_sub = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, lidar_in_topic, 1);
        // image_sub = new message_filters::Subscriber<sensor_msgs::Image>(nh, camera_in_topic, 1);
        // sync = new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), *cloud_sub, *image_sub);
        // sync->setMaxIntervalDuration(ros::Duration(10.0));
        // sync->registerCallback(boost::bind(&camLidarCalib::callback, this, _1, _2));

        // imageSub = n.subscribe(camera_in_topic, 5, &camLidarCalib::imageCallback, this);
        // cloudSub = n.subscribe(lidar_in_topic, 5, &camLidarCalib::cloudCallback, this);
        imageAndCloudSub = n.subscribe(image_and_cloud_in_topic, 5, &camLidarCalib::imageAndCloudCallback, this);
        imagesAndCloudSub = n.subscribe(images_and_cloud_in_topic, 5, &camLidarCalib::imagesAndCloudCallback, this);

        samplingDuration = ros::Duration(0.5);  // in sec
        prevCycleTime = ros::Time(1);

        filtered_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("filtered_cloud_out", 1);
        plane_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("plane_cloud_out", 1);

        projection_matrix = cv::Mat::zeros(3, 3, CV_64F);
        distCoeff = cv::Mat::zeros(5, 1, CV_64F);
        boardDetectedInCam = false;

        tvec = cv::Mat::zeros(3, 1, CV_64F);
        rvec = cv::Mat::zeros(3, 1, CV_64F);
        C_R_W = cv::Mat::eye(3, 3, CV_64F);
        c_R_w = Eigen::Matrix3d::Identity();

        dx = readParam<double>(nh, "dx");
        dy = readParam<double>(nh, "dy");
        checkerboard_rows = readParam<int>(nh, "checkerboard_rows");
        checkerboard_cols = readParam<int>(nh, "checkerboard_cols");
        min_points_on_plane = readParam<int>(nh, "min_points_on_plane");
        max_points_on_plane = readParam<int>(nh, "max_points_on_plane");
        num_views = readParam<int>(nh, "num_views");
        no_of_initializations = readParam<int>(nh, "no_of_initializations");
        initializations_file = readParam<std::string>(nh, "initializations_file");
        for(int i = 0; i < checkerboard_rows; i++)
            for (int j = 0; j < checkerboard_cols; j++)
                object_points.emplace_back(cv::Point3f(i*dx, j*dy, 0.0));

        result_str = readParam<std::string>(nh, "result_file");
        result_rpy = readParam<std::string>(nh, "result_rpy_file");

        cam_config_file_path = readParam<std::string>(nh, "cam_config_file_path");
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
    }

    void readCameraParams(std::string cam_config_file_path,
                          int &image_height,
                          int &image_width,
                          cv::Mat &D,
                          cv::Mat &K) {
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
        if (n.getParam(name, ans))
        {
            ROS_INFO_STREAM("Loaded " << name << ": " << ans);
        }
        else
        {
            ROS_ERROR_STREAM("Failed to load " << name);
            n.shutdown();
        }
        return ans;
    }

    void addGaussianNoise(Eigen::Matrix4d &transformation) {
        std::vector<double> data_rot = {0, 0, 0};
        const double mean_rot = 0.0;
        std::default_random_engine generator_rot;
        generator_rot.seed(std::chrono::system_clock::now().time_since_epoch().count());
        std::normal_distribution<double> dist(mean_rot, 90);

        // Add Gaussian noise
        for (auto& x : data_rot) {
            x = x + dist(generator_rot);
        }

        double roll = data_rot[0]*M_PI/180;
        double pitch = data_rot[1]*M_PI/180;
        double yaw = data_rot[2]*M_PI/180;

        Eigen::Matrix3d m;
        m = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())
            * Eigen::AngleAxisd(pitch,  Eigen::Vector3d::UnitY())
            * Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());

        std::vector<double> data_trans = {0, 0, 0};
        const double mean_trans = 0.0;
        std::default_random_engine generator_trans;
        generator_trans.seed(std::chrono::system_clock::now().time_since_epoch().count());
        std::normal_distribution<double> dist_trans(mean_trans, 0.5);

        // Add Gaussian noise
        for (auto& x : data_trans) {
            x = x + dist_trans(generator_trans);
        }

        Eigen::Vector3d trans;
        trans(0) = data_trans[0];
        trans(1) = data_trans[1];
        trans(2) = data_trans[2];

        Eigen::Matrix4d trans_noise = Eigen::Matrix4d::Identity();
        trans_noise.block(0, 0, 3, 3) = m;
        trans_noise.block(0, 3, 3, 1) = trans;
        transformation = transformation*trans_noise;
    }

    /// basead on the fact that angle = atan2(norm(cross(a,b)),dot(a,b));
    /// return value is in radians
    double angleBetweenVectors(Eigen::Vector3d a, Eigen::Vector3d b) 
    {
        double angle = 0.0;

        angle = std::atan2(a.cross(b).norm(), a.dot(b));

        return angle;
    }


    void filterCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &inputCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &outputCloud)
    {
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


    void extractChessboardCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &inputCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &outputCloud)
    {
        //--- Plane Segmentation
        pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(
                new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(inputCloud));
        pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
        ransac.setDistanceThreshold(ransac_threshold);
        ransac.computeModel();
        std::vector<int> inliers_indicies;
        ransac.getInliers(inliers_indicies);

        pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud<pcl::PointXYZ>(*inputCloud, inliers_indicies, *plane);

        //--- Statistical Outlier Removal
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(plane);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1);
        sor.filter(*outputCloud);
    }


    void cloudHandler_0(const draconis_demo_custom_msgs::ImagePointcloudMsgConstPtr &msg) {

        pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(msg->pointcloud, *in_cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_x(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ >::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ >::Ptr plane_filtered(new pcl::PointCloud<pcl::PointXYZ>);

        /// Pass through filters
        pcl::PassThrough<pcl::PointXYZ> pass_x;
        pass_x.setInputCloud(in_cloud);
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

        /// Plane Segmentation
        pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(
                new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud_filtered_z));
        pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
        ransac.setDistanceThreshold(ransac_threshold);
        ransac.computeModel();
        std::vector<int> inliers_indicies;
        ransac.getInliers(inliers_indicies);
        pcl::copyPointCloud<pcl::PointXYZ>(*cloud_filtered_z, inliers_indicies, *plane);

        /// Statistical Outlier Removal
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(plane);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1);
        sor.filter (*plane_filtered);

        /// Store the points lying in the filtered plane in a vector
        lidar_points.clear();
        for (size_t i = 0; i < plane_filtered->points.size(); i++) {
            double X = plane_filtered->points[i].x;
            double Y = plane_filtered->points[i].y;
            double Z = plane_filtered->points[i].z;
            lidar_points.push_back(Eigen::Vector3d(X, Y, Z));
        }

        ROS_WARN_STREAM("No of planar_pts: " << plane_filtered->points.size());

        //--- Publish the filtered cloud so can see on rviz

        sensor_msgs::PointCloud2 out_filtered_cloud;
        pcl::toROSMsg(*cloud_filtered_z, out_filtered_cloud);
        out_filtered_cloud.header.frame_id = msg->header.frame_id;
        out_filtered_cloud.header.stamp = msg->header.stamp;
        filtered_cloud_pub.publish(out_filtered_cloud);


        //--- Publish the plane cloud so can see on rviz

        sensor_msgs::PointCloud2 out_plane_cloud;
        pcl::toROSMsg(*plane_filtered, out_plane_cloud);
        out_plane_cloud.header.frame_id = msg->header.frame_id;
        out_plane_cloud.header.stamp = msg->header.stamp;
        plane_cloud_pub.publish(out_plane_cloud);
    }


    void cloudHandler_1(const draconis_demo_custom_msgs::ImagePointcloudMsgConstPtr &msg) {

        pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(msg->pointcloud, *in_cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_x(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ >::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ >::Ptr plane_filtered(new pcl::PointCloud<pcl::PointXYZ>);

        /// Pass through filters
        pcl::PassThrough<pcl::PointXYZ> pass_x;
        pass_x.setInputCloud(in_cloud);
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

        //--- Publish the filtered cloud so can see on rviz

        sensor_msgs::PointCloud2 out_filtered_cloud;
        pcl::toROSMsg(*cloud_filtered_z, out_filtered_cloud);
        out_filtered_cloud.header.frame_id = msg->pointcloud.header.frame_id;
        out_filtered_cloud.header.stamp = msg->pointcloud.header.stamp;
        filtered_cloud_pub.publish(out_filtered_cloud);


        //--- Plane Segmentation
        pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(cloud_filtered_z);

        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_plane_vector;
        
        int initNumOfPoints = temp_cloud->size();

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;

        // Optional
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(ransac_threshold);

        // Create extract filter
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        int k = 0;

        while (temp_cloud->size() > 0.1*initNumOfPoints)
        {
            seg.setInputCloud(temp_cloud);
            seg.segment(*inliers, *coefficients);

            size_t numPointsOnPlane = inliers->indices.size ();

            // Create the filtering object
            extract.setInputCloud(temp_cloud);
            extract.setIndices(inliers);
            
            if (!((min_points_on_plane < numPointsOnPlane) && (numPointsOnPlane < max_points_on_plane)))
            {
                extract.setNegative(true);
                extract.filter(*temp_cloud);
                continue;
            }

            ROS_INFO("inside while loop %d, %ld", k, numPointsOnPlane);
            k++;

            // Extract the inliers
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
            extract.setNegative(false);
            extract.filter(*cloud_plane);

            cloud_plane_vector.push_back(cloud_plane);

            if (cloud_plane_vector.size() > 20)
                break;

            // Get the rest of the cloud for next iteration
            extract.setNegative(true);
            extract.filter(*temp_cloud);
        }

        if (cloud_plane_vector.size() == 0)
            return;

        int selected_cloud_index = 0;
        size_t number_of_point_min = cloud_plane_vector[selected_cloud_index]->size();

        for (int i = 1; i < cloud_plane_vector.size(); i++)
        {
            if (cloud_plane_vector[i]->size() < number_of_point_min)
            {
                selected_cloud_index = i;
                number_of_point_min = cloud_plane_vector[i]->size();
            }
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr selected_cloud = cloud_plane_vector[selected_cloud_index];


        /// Store the points lying in the filtered plane in a vector
        lidar_points.clear();
        for (size_t i = 0; i < selected_cloud->points.size(); i++) {
            double X = selected_cloud->points[i].x;
            double Y = selected_cloud->points[i].y;
            double Z = selected_cloud->points[i].z;
            lidar_points.push_back(Eigen::Vector3d(X, Y, Z));
        }

        ROS_WARN_STREAM("No of planar_pts: " << selected_cloud->points.size());


        //--- Publish the plane cloud so can see on rviz

        sensor_msgs::PointCloud2 out_plane_cloud;
        pcl::toROSMsg(*selected_cloud, out_plane_cloud);
        out_plane_cloud.header.frame_id = msg->pointcloud.header.frame_id;
        out_plane_cloud.header.stamp = msg->pointcloud.header.stamp;
        plane_cloud_pub.publish(out_plane_cloud);
    }


    void cloudHandler(const draconis_demo_custom_msgs::ImagePointcloudMsgConstPtr &msg)
    {
        //--- Get input cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(msg->pointcloud, *in_cloud);

        //--- Filter input cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        filterCloud(in_cloud, filtered_cloud);

        //--- Extract chessboard cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr chessboard_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        extractChessboardCloud(filtered_cloud, chessboard_cloud);

        //--- Store the points lying in the filtered plane in a vector
        lidar_points.clear();
        for (size_t i = 0; i < chessboard_cloud->points.size(); i++) {
            double X = chessboard_cloud->points[i].x;
            double Y = chessboard_cloud->points[i].y;
            double Z = chessboard_cloud->points[i].z;
            lidar_points.push_back(Eigen::Vector3d(X, Y, Z));
        }

        //--- Publish the filtered cloud so can see on rviz
        sensor_msgs::PointCloud2 out_filtered_cloud;
        pcl::toROSMsg(*filtered_cloud, out_filtered_cloud);
        out_filtered_cloud.header.frame_id = msg->pointcloud.header.frame_id;
        out_filtered_cloud.header.stamp = msg->pointcloud.header.stamp;
        filtered_cloud_pub.publish(out_filtered_cloud);


        //--- Publish the plane cloud so can see on rviz
        sensor_msgs::PointCloud2 out_plane_cloud;
        pcl::toROSMsg(*chessboard_cloud, out_plane_cloud);
        out_plane_cloud.header.frame_id = msg->pointcloud.header.frame_id;
        out_plane_cloud.header.stamp = msg->pointcloud.header.stamp;
        plane_cloud_pub.publish(out_plane_cloud);

    }


    void imageHandler(const draconis_demo_custom_msgs::ImagePointcloudMsgConstPtr &msg) {
        try {
            boost::shared_ptr<void const> tracked_object;
            image_in = cv_bridge::toCvShare(msg->image, tracked_object, "bgr8")->image;
            boardDetectedInCam = cv::findChessboardCorners(image_in,
                                                           cv::Size(checkerboard_cols, checkerboard_rows),
                                                           image_points,
                                                           cv::CALIB_CB_ADAPTIVE_THRESH+
                                                           cv::CALIB_CB_NORMALIZE_IMAGE);
            // cv::drawChessboardCorners(image_in,
            //                           cv::Size(checkerboard_cols, checkerboard_rows),
            //                           image_points,
            //                           boardDetectedInCam);
            if(image_points.size() == object_points.size())
            {
                cv::solvePnP(object_points, image_points, projection_matrix, distCoeff, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
                projected_points.clear();
                cv::projectPoints(object_points, rvec, tvec, projection_matrix, distCoeff, projected_points, cv::noArray());
                for(int i = 0; i < projected_points.size(); i++){
                    cv::circle(image_in, projected_points[i], 3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA, 0);
                }
                cv::Rodrigues(rvec, C_R_W);
                cv::cv2eigen(C_R_W, c_R_w);
                c_t_w = Eigen::Vector3d(tvec.at<double>(0),
                                        tvec.at<double>(1),
                                        tvec.at<double>(2));

                r3 = c_R_w.block<3,1>(0,2);
                Nc = (r3.dot(c_t_w))*r3;
            }
            cv::resize(image_in, image_resized, cv::Size(), 1.0, 1.0);
            cv::imshow("view", image_resized);
            cv::waitKey(10);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("Could not convert from '%s' to 'bgr8'.",
                      msg->image.encoding.c_str());
        }
    }

    void runSolver() {
        if (lidar_points.size() > min_points_on_plane && boardDetectedInCam) {
            if (r3.dot(r3_old) < 0.90) {
                r3_old = r3;
                all_normals.push_back(Nc);
                all_lidar_points.push_back(lidar_points);
                ROS_ASSERT(all_normals.size() == all_lidar_points.size());
                ROS_INFO_STREAM("Recording View number: " << all_normals.size());
                if (all_normals.size() >= num_views) {
                    ROS_INFO_STREAM("Starting optimization...");
                    init_file.open(initializations_file);
                    for(int counter = 0; counter < no_of_initializations; counter++) {
                        /// Start Optimization here

                        /// Step 1: Initialization
                        Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
                        addGaussianNoise(transformation_matrix);
                        Eigen::Matrix3d Rotn = transformation_matrix.block(0, 0, 3, 3);
                        Eigen::Vector3d axis_angle;
                        ceres::RotationMatrixToAngleAxis(Rotn.data(), axis_angle.data());

                        Eigen::Vector3d Translation = transformation_matrix.block(0, 3, 3, 1);

                        Eigen::Vector3d rpy_init = Rotn.eulerAngles(0, 1, 2)*180/M_PI;
                        Eigen::Vector3d tran_init = transformation_matrix.block(0, 3, 3, 1);

                        Eigen::VectorXd R_t(6);
                        R_t(0) = axis_angle(0);
                        R_t(1) = axis_angle(1);
                        R_t(2) = axis_angle(2);
                        R_t(3) = Translation(0);
                        R_t(4) = Translation(1);
                        R_t(5) = Translation(2);
                        /// Step2: Defining the Loss function (Can be NULL)
//                    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
//                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                        ceres::LossFunction *loss_function = NULL;

                        /// Step 3: Form the Optimization Problem
                        ceres::Problem problem;
                        problem.AddParameterBlock(R_t.data(), 6);
                        for (int i = 0; i < all_normals.size(); i++) {
                            Eigen::Vector3d normal_i = all_normals[i];
                            std::vector<Eigen::Vector3d> lidar_points_i
                                    = all_lidar_points[i];
                            for (int j = 0; j < lidar_points_i.size(); j++) {
                                Eigen::Vector3d lidar_point = lidar_points_i[j];
                                ceres::CostFunction *cost_function = new
                                        ceres::AutoDiffCostFunction<CalibrationErrorTerm, 1, 6>
                                        (new CalibrationErrorTerm(lidar_point, normal_i));
                                problem.AddResidualBlock(cost_function, loss_function, R_t.data());
                            }
                        }

                        /// Step 4: Solve it
                        ceres::Solver::Options options;
                        options.max_num_iterations = 200;
                        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                        options.minimizer_progress_to_stdout = false;
                        ceres::Solver::Summary summary;
                        ceres::Solve(options, &problem, &summary);
//                        std::cout << summary.FullReport() << '\n';



                        /// Printing and Storing C_T_L in a file
                        ceres::AngleAxisToRotationMatrix(R_t.data(), Rotn.data());
                        Eigen::MatrixXd C_T_L(3, 4);
                        C_T_L.block(0, 0, 3, 3) = Rotn;
                        C_T_L.block(0, 3, 3, 1) = Eigen::Vector3d(R_t[3], R_t[4], R_t[5]);
                        std::cout << "RPY = " << Rotn.eulerAngles(0, 1, 2)*180/M_PI << std::endl;
                        std::cout << "t = " << C_T_L.block(0, 3, 3, 1) << std::endl;

                        init_file << rpy_init(0) << "," << rpy_init(1) << "," << rpy_init(2) << ","
                                  << tran_init(0) << "," << tran_init(1) << "," << tran_init(2) << "\n";
                        init_file << Rotn.eulerAngles(0, 1, 2)(0)*180/M_PI << "," << Rotn.eulerAngles(0, 1, 2)(1)*180/M_PI << "," << Rotn.eulerAngles(0, 1, 2)(2)*180/M_PI << ","
                                  << R_t[3] << "," << R_t[4] << "," << R_t[5] << "\n";

                        /// Step 5: Covariance Estimation
                        ceres::Covariance::Options options_cov;
                        ceres::Covariance covariance(options_cov);
                        std::vector<std::pair<const double*, const double*> > covariance_blocks;
                        covariance_blocks.push_back(std::make_pair(R_t.data(), R_t.data()));
                        CHECK(covariance.Compute(covariance_blocks, &problem));
                        double covariance_xx[6 * 6];
                        covariance.GetCovarianceBlock(R_t.data(),
                                                      R_t.data(),
                                                      covariance_xx);

                        Eigen::MatrixXd cov_mat_RotTrans(6, 6);
                        cv::Mat cov_mat_cv = cv::Mat(6, 6, CV_64F, &covariance_xx);
                        cv::cv2eigen(cov_mat_cv, cov_mat_RotTrans);

                        Eigen::MatrixXd cov_mat_TransRot(6, 6);
                        cov_mat_TransRot.block(0, 0, 3, 3) = cov_mat_RotTrans.block(3, 3, 3, 3);
                        cov_mat_TransRot.block(3, 3, 3, 3) = cov_mat_RotTrans.block(0, 0, 3, 3);
                        cov_mat_TransRot.block(0, 3, 3, 3) = cov_mat_RotTrans.block(3, 0, 3, 3);
                        cov_mat_TransRot.block(3, 0, 3, 3) = cov_mat_RotTrans.block(0, 3, 3, 3);

                        double  sigma_xx = sqrt(cov_mat_TransRot(0, 0));
                        double  sigma_yy = sqrt(cov_mat_TransRot(1, 1));
                        double  sigma_zz = sqrt(cov_mat_TransRot(2, 2));

                        double sigma_rot_xx = sqrt(cov_mat_TransRot(3, 3));
                        double sigma_rot_yy = sqrt(cov_mat_TransRot(4, 4));
                        double sigma_rot_zz = sqrt(cov_mat_TransRot(5, 5));

                        std::cout << "sigma_xx = " << sigma_xx << "\t"
                                  << "sigma_yy = " << sigma_yy << "\t"
                                  << "sigma_zz = " << sigma_zz << std::endl;

                        std::cout << "sigma_rot_xx = " << sigma_rot_xx*180/M_PI << "\t"
                                  << "sigma_rot_yy = " << sigma_rot_yy*180/M_PI << "\t"
                                  << "sigma_rot_zz = " << sigma_rot_zz*180/M_PI << std::endl;

                        std::ofstream results;
                        results.open(result_str);
                        results << C_T_L;
                        results.close();

                        std::ofstream results_rpy;
                        results_rpy.open(result_rpy);
                        results_rpy << Rotn.eulerAngles(0, 1, 2)*180/M_PI << "\n" << C_T_L.block(0, 3, 3, 1);
                        results_rpy.close();

                        ROS_INFO_STREAM("No of initialization: " << counter);
                    }
                    init_file.close();
                    ros::shutdown();

                }
            } else {
                ROS_WARN_STREAM("Not enough Rotation, view not recorded");
            }
        } else {
            if(!boardDetectedInCam)
                ROS_WARN_STREAM("Checker-board not detected in Image.");
            else {
                ROS_WARN_STREAM("Checker Board Detected in Image?: " << boardDetectedInCam << "\t" <<
                "No of LiDAR pts: " << lidar_points.size() << " (Check if this is less than threshold) ");
            }
        }
    }

    void callback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
                  const sensor_msgs::ImageConstPtr &image_msg) 
    {
        // ROS_INFO("Hi");
        // imageHandler(image_msg);
        // cloudHandler(cloud_msg);
        // runSolver();
    }

    void imageAndCloudCallback(const draconis_demo_custom_msgs::ImagePointcloudMsgConstPtr &msg)
    {
        // ROS_INFO("Hi");
        imageHandler(msg);

        if (!boardDetectedInCam)
            return;

        cloudHandler(msg);
        runSolver();
    }

    void imagesAndCloudCallback(const draconis_demo_custom_msgs::ImagesAndPointcloudMsgConstPtr &msg)
    {
        // ROS_INFO("Hi");
        imageHandler(msg->image_left);

        if (!boardDetectedInCam)
            return;

        cloudHandler(msg->pointcloud);
        runSolver();
    }

    void imageCallback(const sensor_msgs::ImageConstPtr &image_msg)
    {
        // ROS_INFO("image %0.2f", image_msg->header.stamp.toSec());

        // ros::Duration period = ros::Time::now() - prevCycleTime;

        // if (period < samplingDuration)
        //     return;

        // prevCycleTime = ros::Time::now();


        // if (cloudMsgPtr == NULL)
        //     return;

        // imageHandler(image_msg);
        // cloudHandler(cloudMsgPtr);
        // runSolver();
    }

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
    {
        // ROS_INFO("cloud %0.2f", cloud_msg->header.stamp.toSec());
        // cloudMsgPtr = cloud_msg;
    }
};

int main(int argc, char** argv) {
    ROS_INFO_STREAM("Start Camera Lidar Calibration!");
    ros::init(argc, argv, "CameraLidarCalib_node");
    ros::NodeHandle nh("~");
    camLidarCalib cLC(nh);
    ros::spin();
    return 0;
}
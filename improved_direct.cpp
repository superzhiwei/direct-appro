#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <ctime>
#include <climits>
#include <string>

#include <opencv2/opencv.hpp>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;
using namespace cv;
using namespace g2o;
using namespace Eigen;


//本程序演示了两两之间的稀疏直接法，与课本上的演示略有差别

struct Measurement
{
    Measurement(Eigen::Vector3d p, float g):pos_world(p),grayscale(g){}
    Eigen::Vector3d pos_world;
    float grayscale;
};

inline Eigen::Vector3d project2Dto3D(int x, int y, int d,float fx,float fy, float cx, float cy, float depth_scale)
{
    float zz = float(d)/depth_scale;
    float yy = (float(y) - cy)*zz/fy;
    float xx = (float(x) - cx)*zz/fx;
    return Eigen::Vector3d(xx,yy,zz);
}

inline Eigen::Vector2d project3Dto2D(float x, float y,float z,float fx, float fy, float cx, float cy)
{
    float u = fx*x/z +cx;
    float v = fy*y/z +cy;
    return Eigen::Vector2d(u,v);
}

bool poseEstimationDirect(const vector<Measurement>& measurements,cv::Mat* gray,Eigen::Matrix3d& K,Eigen::Isometry3d& T_c_r);

class EdgeSE3ProjectDirect: public g2o::BaseUnaryEdge<1,double ,g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectDirect(){}

    EdgeSE3ProjectDirect(Eigen::Vector3d point, float fx, float fy, float cx, float cy,
    cv::Mat* img):x_world(point),fx_(fx),fy_(fy),cx_(cx),cy_(cy),image_(img) {}

    virtual void computeError()
    {
        const VertexSE3Expmap* v = static_cast<VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d x_local = v->estimate().map(x_world);
        float x = x_local[0]*fx_/x_local[2] + cx_;
        float y = x_local[1]*fy_/x_local[2] + cy_;

        if (x<4||(x+4)>image_->cols ||y<4 ||(y+4)>image_->rows)
        {
            _error(0,0) = 0.0;
            this->setLevel(1);
        } else
        {
            _error(0,0) = getPixelValue(x,y) - _measurement;
        }

    }

    virtual void linearizeOplus()
    {

        if (level() == 1)
        {
            _jacobianOplusXi = Eigen::Matrix<double,1,6>::Zero();
            return;
        }

        const VertexSE3Expmap* v1 = static_cast<VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d x_trans = v1->estimate().map(x_world);

        double x = x_trans[0];
        double y = x_trans[1];
        double z = x_trans[2];
        double z2 = z*z;

        double u = fx_*x/z +cx_;
        double v = fy_*y/z +cy_;

        Eigen::Matrix<double ,2,6> jacobian_uv_sigam;
        jacobian_uv_sigam(0,0) = -fx_*x*y/z2;
        jacobian_uv_sigam(0,1) = fx_ + fx_*x*x/z2;
        jacobian_uv_sigam(0,2) = -fx_*y/z;
        jacobian_uv_sigam(0,3) = fx_/z;
        jacobian_uv_sigam(0,4) = 0;
        jacobian_uv_sigam(0,5) = -fx_*x/z2;

        jacobian_uv_sigam(1,0) = -fy_*(1+y*y/z2);
        jacobian_uv_sigam(1,1) = fy_*x*y/z2;
        jacobian_uv_sigam(1,2) = fy_*x/z;
        jacobian_uv_sigam(1,3) = 0;
        jacobian_uv_sigam(1,4) = fy_/z;
        jacobian_uv_sigam(1,5) = -fy_*y/z2;

        Eigen::Matrix<double ,1,2> jacobian_pixel_uv;
        jacobian_pixel_uv(0,0) = (getPixelValue(u+1,v)-getPixelValue(u-1,v))/2;
        jacobian_pixel_uv(0,1) = (getPixelValue(u,v+1)-getPixelValue(u,v-1))/2;

        _jacobianOplusXi = jacobian_pixel_uv*jacobian_uv_sigam;
    }

    virtual bool read(std::istream& in){}
    virtual bool write(std::ostream& out)const {}

protected:
    inline float getPixelValue(float x, float y)
    {
        uchar* data_ptr = &image_->data[int(y) * image_->step + int(x)];
        uchar* data = data_ptr;

        float xx = x - floor(x);
        float yy = y - floor(y);

        return float(
                (1-xx)*(1-yy)*data[0]+
                xx*(1-yy)*data[1]+
                (1-xx)*yy*data[image_->step]+
                xx*yy*data[image_->step+1]
                );
    }

public:
    Eigen::Vector3d x_world;
    float cx_ = 0,cy_ = 0,fx_ = 0,fy_ = 0;
    cv::Mat* image_ = nullptr;

};


int  main()
{
    string path_data = "../data";
    string associate_data = path_data + "/associate.txt";

    fstream direct_file(associate_data,ios_base::in);
    if (!direct_file.is_open())
    {
        cout<<"associate.txt file filed"<<endl;
        return -1;
    }

    cout<<"associate.txt file successed!"<<endl;
    string rgb_file,rgb_time,depth_file,depth_time;

    cv::Mat ref_color,ref_depth,ref_gray,curr_color,curr_depth,curr_gray;
    vector<Measurement> measurements;

    float fx = 325.5;
    float fy = 253.5;
    float cx = 518.0;
    float cy = 519.0;
    float depth_scale = 1000.0;
    Eigen::Matrix3d K;
    K<<fx,0,cx,0,fy,cy,0,0,1;

    Eigen::Isometry3d T_c_r = Eigen::Isometry3d::Identity();
    cv::Mat prve_color;

    for (int index = 0; index < 10; ++index)
    {
        cout<<"********* "<<index+1<<" ***********"<<endl;
        direct_file>>rgb_time>>rgb_file>>depth_time>>depth_file;
        curr_color = cv::imread(path_data+"/"+rgb_file);
        curr_depth = cv::imread(path_data+"/"+depth_file,-1);
        if (curr_color.data == nullptr || curr_depth.data == nullptr)
            continue;

        cv::cvtColor(curr_color,curr_gray,cv::COLOR_BGR2GRAY);

        if (index == 0)
        {
            ref_color = curr_color.clone();
            ref_depth = curr_depth.clone();
            ref_gray = curr_gray.clone();
            vector<cv::KeyPoint> keypoints;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect(ref_color,keypoints);
            for (auto kp:keypoints)
            {
                if (kp.pt.x<20||(kp.pt.x+20)>ref_color.cols || kp.pt.y<20 ||(kp.pt.y+20)>ref_color.rows)
                    continue;
                ushort d = ref_depth.ptr<ushort >(cvRound(kp.pt.y))[cvRound(kp.pt.x)];
                if (d ==0)
                {
                    continue;
                }
                Eigen::Vector3d p3d = project2Dto3D(kp.pt.x,kp.pt.y,d,fx,fy,cx,cy,depth_scale);
                float grayvalue = float(ref_gray.ptr<uchar>(cvRound(kp.pt.y))[cvRound(kp.pt.x)]);

                measurements.push_back(Measurement(p3d,grayvalue));
            }
            continue;
        }


        poseEstimationDirect(measurements,&curr_gray,K,T_c_r);

        cout<<"T_c_r :\n"<<T_c_r.matrix()<<endl;

        T_c_r = Eigen::Isometry3d::Identity();
        cv::Mat img_show ( curr_color.rows*2, curr_color.cols, CV_8UC3 );
        ref_color.copyTo ( img_show ( cv::Rect ( 0,0,curr_color.cols, curr_color.rows ) ) );
        curr_color.copyTo ( img_show ( cv::Rect ( 0,curr_color.rows,curr_color.cols, curr_color.rows ) ) );
        for (auto m:measurements )
        {
            if ( rand() > RAND_MAX/5 )
                continue;
            Eigen::Vector3d p = m.pos_world;
            Eigen::Vector2d pixel_prev = project3Dto2D ( p ( 0,0 ), p ( 1,0 ), p ( 2,0 ), fx, fy, cx, cy );
            Eigen::Vector3d p2 = T_c_r*m.pos_world;
            Eigen::Vector2d pixel_now = project3Dto2D ( p2 ( 0,0 ), p2 ( 1,0 ), p2 ( 2,0 ), fx, fy, cx, cy );
            if ( pixel_now(0,0)<0 || pixel_now(0,0)>=curr_color.cols || pixel_now(1,0)<0 || pixel_now(1,0)>=curr_color.rows )
                continue;

            float b = 255*float ( rand() ) /RAND_MAX;
            float g = 255*float ( rand() ) /RAND_MAX;
            float r = 255*float ( rand() ) /RAND_MAX;
            cv::circle ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), 8, cv::Scalar ( b,g,r ), 2 );
            cv::circle ( img_show, cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +curr_color.rows ), 8, cv::Scalar ( b,g,r ), 2 );
            cv::line ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +curr_color.rows ), cv::Scalar ( b,g,r ), 1 );
        }
        cv::imshow ( "result", img_show );
        cv::waitKey ( 0 );

        measurements.clear();
        ref_color = curr_color.clone();
        ref_depth = curr_depth.clone();
        ref_gray = curr_gray.clone();
        vector<cv::KeyPoint> keypoints;
        cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
        detector->detect(ref_color,keypoints);
        for (auto kp:keypoints)
        {
            if (kp.pt.x<20||(kp.pt.x+20)>ref_color.cols || kp.pt.y<20 ||(kp.pt.y+20)>ref_color.rows)
                continue;
            ushort d = ref_depth.ptr<ushort >(cvRound(kp.pt.y))[cvRound(kp.pt.x)];
            if (d ==0)
            {
                continue;
            }
            Eigen::Vector3d p3d = project2Dto3D(kp.pt.x,kp.pt.y,d,fx,fy,cx,cy,depth_scale);
            float grayvalue = float(ref_gray.ptr<uchar>(cvRound(kp.pt.y))[cvRound(kp.pt.x)]);

            measurements.push_back(Measurement(p3d,grayvalue));
        }

    }

    direct_file.close();
    return 0;

}

bool poseEstimationDirect(const vector<Measurement>& measurements,cv::Mat* gray,Eigen::Matrix3d& K,Eigen::Isometry3d& T_c_r)
{

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;
    DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();
    DirectBlock* solver_ptr = new DirectBlock(unique_ptr<DirectBlock::LinearSolverType>(linearSolver));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<DirectBlock>(solver_ptr));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setEstimate(g2o::SE3Quat(T_c_r.rotation(),T_c_r.translation()));
    pose->setId(0);
    optimizer.addVertex(pose);

    for (int index = 1; index < measurements.size(); ++index)
    {
        EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect(measurements[index-1].pos_world,K(0,0),K(1,1),K(0,2),K(1,2),gray);

        edge->setVertex(0,pose);
        edge->setMeasurement(measurements[index-1].grayscale);
        edge->setInformation(Eigen::Matrix<double ,1,1>::Identity());
        edge->setId(index);

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(1.0);
        edge->setRobustKernel(rk);

        optimizer.addEdge(edge);
    }


    cout<<"edge size :"<<optimizer.edges().size()<<endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);
    T_c_r = pose->estimate();

}
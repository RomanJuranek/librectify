#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "config.h"
#include "liblgroup.h"
#include "geometry.h"
#include "transform.h"


using namespace std;
using namespace Eigen;


namespace librectify {


Vector3f normalize_point(const Vector3f & p)
{
    if (abs(p.z()) < EPS)
        return Vector3f(p.x(), p.y(), 0);
    else
        return Vector3f(p.x()/p.z(), p.y()/p.z(), 1);
}


VanishingPoint::VanishingPoint(const MatrixX3f & h, const VectorXf & w)
{
    Matrix3f cov = h.adjoint() * w.asDiagonal() * h;
    SelfAdjointEigenSolver<Matrix3f> eig(cov);
    Index k;
    eig.eigenvalues().minCoeff(&k);
    coords = normalize_point(eig.eigenvectors().col(k));
    residual = eig.eigenvalues().coeff(k);
    n_inliers = int(w.size());
    support = w.sum();
    #if LGROUP_DEBUG_PRINTS
    clog << "VP: coords=" << RowVector3f(coords) << ", residual=" << residual << ", support=" << support << endl;
    #endif
}


vector<VanishingPoint> fit_vanishing_points(const vector<LineSegment> & lines)
{
    ArrayXi groups = group_id(lines);

    Vector4f bb = bounding_box(lines);
    Vector2f bb_size = bb.tail<2>() - bb.head<2>();
    Vector2f center = bb.head<2>() + 0.5 * bb_size;
    float scale = bb_size.minCoeff();
    float _x = center.x();
    float _y = center.y();

    #if LGROUP_DEBUG_PRINTS
    clog << "bb:\n" << bb << endl;
    clog << "scale=" << scale << ", center = " << _x << "," << _y << endl;
    #endif

    vector<LineSegment> normlized_lines(lines);
    for (size_t i=0; i<lines.size(); ++i)
    {
        LineSegment & l = normlized_lines[i];
        l.x1 = (l.x1 - _x)/scale;
        l.y1 = (l.y1 - _y)/scale;
        l.x2 = (l.x2 - _x)/scale;
        l.y2 = (l.y2 - _y)/scale;
    }

    MatrixX3f h = homogeneous(normlized_lines).rowwise().normalized();
    VectorXf wts = weigths(normlized_lines);
    VectorXf len = length(normlized_lines);

    int n_groups = groups.maxCoeff() + 1;

    vector<VanishingPoint> res;
    
    for (int g = 0; g < n_groups; ++g)
    {
        //cout << g << ": " << (groups == g).count() << endl;
        auto idx = index_array(groups == g);
        res.emplace_back(VanishingPoint(h(idx,all), wts(idx).cwiseProduct(len(idx))  ));
    }

    for (size_t i = 0; i < res.size(); ++i)
    {
        if (res[i].coords.z() > 0)
        {
            res[i].coords.x() = (res[i].coords.x() * scale) + center.x();
            res[i].coords.y() = (res[i].coords.y() * scale) + center.y();
        }
    }

    return res;
}


Matrix<float,4,3> compute_image_transform(
    int width, int height,
    const Vector3f & horizontal_vp,
    const Vector3f & vertical_vp
    )
{
    auto vanishing_line = horizontal_vp.cross(vertical_vp);
    Matrix3f H = Matrix3f::Identity();
    H.row(2) = vanishing_line.array() / vanishing_line(2);

    Vector3f vp_h_post = (H * horizontal_vp);
    Vector3f vp_v_post = (H * vertical_vp);

   if (vp_h_post(0) < 0)
       vp_h_post = -vp_h_post;

   if (vp_v_post(1) < 0)
       vp_v_post = -vp_v_post;

    Matrix3f A1 = Matrix3f::Identity();
    A1.col(0).head(2) = vp_h_post.head(2).normalized();
    A1.col(1).head(2) = vp_v_post.head(2).normalized();
    auto A = A1.inverse();

    Matrix3f M = A * H;
    #if LGROUP_DEBUG_PRINTS
    clog << "compute_image_transform: M=\n" << M << endl; 
    clog << "compute_image_transform: det M = " << M.determinant() << endl;
    #endif


    Matrix<float,3,4> coords;
    coords.row(0) = Array4i(0,width,width,0).cast<float>() - float(width)/2;
    coords.row(1) = Array4i(0,0,height,height).cast<float>() - float(height)/2;
    coords.row(2) = Vector4f::Ones();

    Matrix<float,3,4> warped_coords = M * coords;
    auto scale = (1.0/warped_coords.row(2).array()).matrix().asDiagonal();
    warped_coords = warped_coords * scale;

    warped_coords.row(0).array() += float(width)/2;
    warped_coords.row(1).array() += float(height)/2;

    #if LGROUP_DEBUG_PRINTS
    clog << "compute_image_transform: coords=\n" << coords << endl << endl;
    clog << "compute_image_transform: warped=\n" << warped_coords << endl << endl;
    #endif

    return warped_coords.transpose(); // coordinates in rows
}


VanishingPoint select_vertical_point(
    const vector<VanishingPoint> & vps,
    const Vector3f & center,
    float angular_tolerance,
    float min_distance)
{
    float cos_threshold = cos(angular_tolerance/180.0f * float(M_PI));
    #if LGROUP_DEBUG_PRINTS
    clog << "select_vertical_point: min distance=" << min_distance << endl;
    clog << "select_vertical_point: threshold=" << cos_threshold << endl;
    #endif
    for (size_t i = 0; i < vps.size(); ++i)
    {
        auto & v = vps[i];
        bool angular_filter = abs((v.direction(center).adjoint() * Vector2f(0,1))) > cos_threshold;
        bool distance_filter = v.distance(center) > min_distance;
        #if LGROUP_DEBUG_PRINTS
        clog << "select_vertical_point: coords=" << RowVector3f(v.coords)
                << ", angle score=" << abs((v.direction(center).adjoint() * Vector2f(0,1)))
                << ", distance=" << v.distance(center) << endl;
        #endif
        if (angular_filter && distance_filter)
        {
            #if LGROUP_DEBUG_PRINTS
            clog << "Selected" << endl;
            #endif
            return v;
        }
    }
    Vector3f h {0,1,0};
    #if LGROUP_DEBUG_PRINTS
    clog << "select_vertical_point: All rejected, using " << RowVector3f(h) << endl;
    #endif
    return VanishingPoint(h);
}


VanishingPoint select_horizontal_point(
    const vector<VanishingPoint> & vps,
    const Vector3f & center,
    const VanishingPoint & vertical,
    float min_distance)
{
    Vector2f vd = vertical.direction(center);
    #if LGROUP_DEBUG_PRINTS
    clog << "select_horizontal_point: min distance=" << min_distance << endl;
    #endif

    for (size_t i = 0; i < vps.size(); ++i)
    {
        auto & v = vps[i];
        if (v.coords == vertical.coords)
        {
            continue; // Avoid selection of the same point as vertical
        }
        float angle_score = v.direction(center).adjoint() * vd;
        bool horizon_filter = angle_score < 0.1 && angle_score > -0.7;
        bool distance_filter = v.distance(center) > min_distance;
        #if LGROUP_DEBUG_PRINTS
        clog << "select_horizontal_point: Selected: coords=" << RowVector3f(v.coords)
             << ", horizon_score=" << (v.direction(center).adjoint() * vd).eval()
             << ", distance=" << v.distance(center) << endl;
        #endif
        if (horizon_filter && distance_filter)
        {
            #if LGROUP_DEBUG_PRINTS
            clog << "Selected" << endl;
            #endif
            return v;
        }
    }
    Vector3f h {1, 0, 0};
    #if LGROUP_DEBUG_PRINTS
    clog << "select_horizontal_point: All rejected, using " << RowVector3f(h) << endl;
    #endif
    return VanishingPoint(h); // orthogonal to vertical point
}

} // namespace
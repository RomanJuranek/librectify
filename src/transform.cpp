#include <iostream>
#include <set>
#include <map>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "config.h"
#include "librectify.h"
#include "line_pencil.h"
#include "geometry.h"
#include "transform.h"
#include "utils.h"


using namespace std;
using namespace Eigen;


namespace librectify {


Vector3f fit_single_vanishing_points(const vector<LineSegment> & lines, int g)
{
    auto bb = bounding_box(lines);
    auto c = bbox_center(bb);
    float scale = bbox_size(bb).maxCoeff();
    auto lines_norm = normalize_lines(lines, c, scale);

    LinePencilModel model(lines_norm);

    ArrayXi groups = group_id(lines);
    ArrayXi indices = ArrayXi();
    if (g > 0)
        indices = nonzero(groups == g);

    auto vp = normalize_point(model.fit_optimal(indices));
    // Apply un-normalization to get coords in original space
    if (vp.z() > 0)
    {
        vp.x() = scale * vp.x() + c.x(); // can be done with matrix multiplication
        vp.y() = scale * vp.y() + c.y();
    }

    return vp;
}




map<int, Vector3f> fit_vanishing_points(const vector<LineSegment> & lines)
{
    auto bb = bounding_box(lines);
    auto c = bbox_center(bb);
    float scale = bbox_size(bb).maxCoeff();
    auto lines_norm = normalize_lines(lines, c, scale);

    LinePencilModel model(lines_norm);

    ArrayXi groups = group_id(lines);
    
    set<int> unique_groups(groups.begin(), groups.end());
    unique_groups.erase(-1); // remove group -1 which is ignored

    map<int, Vector3f> res;

    for (auto g : unique_groups)
    {
        auto indices = nonzero(groups == g);
        auto vp = normalize_point(model.fit_optimal(indices));
        // Apply un-normalization to get coords in original space
        if (vp.z() > 0)
        {
            vp.x() = scale * vp.x() + c.x(); // can be done with matrix multiplication
            vp.y() = scale * vp.y() + c.y();
        }
        res[g] = vp;
    }
    return res;
}


CornerTransform compute_image_transform(
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


Vector3f select_vertical_point(
    const MatrixX3f & vps,
    const Vector3f & center,
    float angular_tolerance,
    float min_distance)
{
    float cos_threshold = cos(angular_tolerance/180.0f * float(M_PI));
    #if LGROUP_DEBUG_PRINTS
    clog << "select_vertical_point: min distance=" << min_distance << endl;
    clog << "select_vertical_point: threshold=" << cos_threshold << endl;
    #endif
    for (Index i = 0; i < vps.rows(); ++i)
    {
        auto & v = vps.row(i);
        bool angular_filter = abs((direction(v,center).adjoint() * Vector2f(0,1))) > cos_threshold;
        bool distance_filter = distance(v,center) > min_distance;
        #if LGROUP_DEBUG_PRINTS
        clog << "select_vertical_point: coords=" << RowVector3f(v)
                << ", angle score=" << abs((direction(v,center).adjoint() * Vector2f(0,1)))
                << ", distance=" << distance(v,center) << endl;
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
    return h;
}


Vector3f select_horizontal_point(
    const MatrixX3f & vps,
    const Vector3f & center,
    const Vector3f & vertical,
    float min_distance)
{
    Vector2f vd = direction(vertical, center);
    #if LGROUP_DEBUG_PRINTS
    clog << "select_horizontal_point: min distance=" << min_distance << endl;
    #endif

    for (Index i = 0; i < vps.rows(); ++i)
    {
        auto & v = vps.row(i);
        if (Vector3f(v) == vertical)
        {
            continue; // Avoid selection of the same point as vertical
        }
        float angle_score = direction(v, center).adjoint() * vd;
        bool horizon_filter = angle_score < 0.05 && angle_score > -0.7;
        bool distance_filter = distance(v, center) > min_distance;
        #if LGROUP_DEBUG_PRINTS
        clog << "select_horizontal_point: Selected: coords=" << RowVector3f(v)
             << ", horizon_score=" << (direction(v,center).adjoint() * vd).eval()
             << ", distance=" << distance(v,center) << endl;
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
    return h; // orthogonal to vertical point
}

} // namespace
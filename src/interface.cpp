#include <iostream>

#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "config.h"
#include "librectify.h"
#include "image.h"
#include "line_detector.h"
#include "line_pencil.h"
#include "geometry.h"
#include "transform.h"
#include "utils.h"
#include "threading.h"


using namespace Eigen;
using namespace std;


namespace librectify {


template<class InputIterator, class OutputIterator>
static void _filter_lines(InputIterator first, InputIterator last, OutputIterator dst, float min_length)
{
    min_length = max(min_length, LINE_MIN_LENGTH);
    vector<LineSegment> filtered;
    copy_if(first, last, dst, [&](const LineSegment&x) { return length(x)>min_length && x.err < LINE_MAX_ERR;} );
}


LineSegment * find_line_segment_groups(
    InputPixelType * buffer, int width, int height, int stride,
    float min_length, // filtering
    bool refine,
    int num_threads,
    int * n_lines 
    )
{
    ThreadContext ctx(num_threads);

    // Init image from buffer
    Image im = image_from_buffer(buffer, width, height, stride);
    // Detect the lines
    auto lines = find_line_segments(im, SEED_DIST, SEED_RATIO, TRACE_TOLERANCE, ctx);

    if (refine)
    {
        lines = postprocess_lines_segments(lines, ctx);
    }

    vector<LineSegment> filtered;
    filtered.reserve(lines.size());
    _filter_lines(lines.begin(), lines.end(), back_inserter(filtered), min_length);

    estimate_line_pencils(filtered, MAX_MODELS, ESTIMATOR_INLIER_MAX_ANGLE_DEG, ESTIMATOR_GARBAGE_MAX_ANGLE_DEG, ctx);
    auto & groupped = filtered;
    
    // Construct resulting array and copy data
    LineSegment * res = new LineSegment[groupped.size()];
    copy_n(groupped.begin(), groupped.size(), res);
    *n_lines = int(groupped.size());

    return res;
}


Point point_from_vector(const Vector3f & p)
{
    return Point {p.x(), p.y(), p.z()};
}

Vector3f vector_from_point(const Point & p)
{
    return Vector3f {p.x, p.y, p.z};
}

ImageTransform compute_rectification_transform_from_vp(
    int width, int height,
    const Point & vp_h, const Point & vp_v)
{
    Vector3f image_center{ float(width) / 2, float(height) / 2, 0 };

    Vector3f v1 = vector_from_point(vp_h);
    if (v1.z() != 0)
        v1.head(2) -= image_center.head(2);
    Vector3f v2 = vector_from_point(vp_v);
    if (v2.z() != 0)
        v2.head(2) -= image_center.head(2);

    auto tform = compute_image_transform(width, height, v1, v2);
    
    ImageTransform T;
    T.width = width;
    T.height = height;
    T.top_left = point_from_vector(tform.row(0));
    T.top_right = point_from_vector(tform.row(1));
    T.bottom_right = point_from_vector(tform.row(2));
    T.bottom_left = point_from_vector(tform.row(3));
    T.horizontal_vp = vp_h;
    T.vertical_vp = vp_v;

    return T;
}


ImageTransform compute_rectification_transform(
    LineSegment * lines, int n_lines,
    int width, int height,
    const RectificationConfig & cfg)
{
    vector<LineSegment> groupped(lines, lines+n_lines);

    auto group_vp_map = fit_vanishing_points(groupped);
    MatrixX3f vps(group_vp_map.size(), 3);
    int k = 0;
    for (const auto & v: group_vp_map)
    {
        vps.row(k) = v.second;
        ++k;
    }

    Vector3f image_center {float(width)/2, float(height)/2, 1};

    float diagonal_size = image_center.head(2).norm();

    float min_v_distance = max(cfg.vertical_vp_min_distance, 1.0f) * diagonal_size;
    auto vp_v = select_vertical_point(vps, image_center, cfg.vertical_vp_angular_tolerance, min_v_distance);
    
    float min_h_distance = max(cfg.horizontal_vp_min_distance, 1.0f) * diagonal_size;
    auto vp_h = select_horizontal_point(vps, image_center, vp_v, min_h_distance);

    Vector3f v1 = vp_h;
    if (v1.z() != 0)
        v1.head(2) -= image_center.head(2);
    Vector3f v2 = vp_v;
    if (v2.z() != 0)
        v2.head(2) -= image_center.head(2);

    Vector3f v1_hat = v1;
    switch (cfg.h_strategy)
    {
        case ROTATE_H:
            v1_hat.z() = 0;
            break;
        case ROTATE_V:
            v1_hat = {-v2.y(), v2.x(), 0};
            break;
        case RECTIFY:
            break;
        case KEEP:
        default:
            v1_hat = {1,0,0};
            break;
    }

    Vector3f v2_hat = v2;
    switch (cfg.v_strategy)
    {
        case ROTATE_H:
            v2_hat = {-v1.y(), v1.x(), 0};
            break;
        case ROTATE_V:
            v2_hat.z() = 0;
            break;
        case RECTIFY:
            break;
        case KEEP:
        default:
            v2_hat = {0,1,0};
            break;
    }
    
    auto tform = compute_image_transform(width, height, v1_hat, v2_hat);
    
    ImageTransform T;
    T.width = width;
    T.height = height;
    
    if (v1_hat.z() != 0)
        v1_hat.head(2) += image_center.head(2);
    if (v2_hat.z() != 0)
        v2_hat.head(2) += image_center.head(2);

    T.top_left = point_from_vector(tform.row(0));
    T.top_right = point_from_vector(tform.row(1));
    T.bottom_right = point_from_vector(tform.row(2));
    T.bottom_left = point_from_vector(tform.row(3));
    T.horizontal_vp = point_from_vector(v1_hat);
    T.vertical_vp = point_from_vector(v2_hat);

    return T;
}


Point fit_vanishing_point(const LineSegment * lines_array, int n_lines, int group)
{
    vector<LineSegment> lines(lines_array, lines_array+n_lines);
    return point_from_vector(fit_single_vanishing_points(lines, group));
}


void assign_to_group(
    const LineSegment * lines_array, int n_lines,
    LineSegment * new_lines_array, int n_new_lines,
    float angular_tolarance)
{
    vector<LineSegment> lines(lines_array, lines_array+n_lines);
    auto group_to_vp_map = fit_vanishing_points(lines);

    vector<LineSegment> new_lines(new_lines_array, new_lines_array+n_new_lines);
    MatrixX2f a = anchor_point(new_lines);
    MatrixX2f d = direction_vector(new_lines).rowwise().normalized();
    
    ArrayXf min_inclination = ArrayXf::Zero(n_new_lines);
    ArrayXi min_index = ArrayXi::Zero(n_new_lines);

    // cout << "size: " << new_lines.size() << endl;
    // cout << "a=\n" << a << endl;
    // cout << "d=\n" << d << endl;

    for (const auto & vp : group_to_vp_map)
    {
        Vector3f p = vp.second;
        ArrayXf x = inclination(a, d, p).array();

        // cout << "p: " << RowVector3f(p) << endl;
        // cout << "x: " << x.transpose() << endl;
        
        Array<bool,-1,1> mask = (x > min_inclination).eval();
        min_inclination = mask.select(x, min_inclination);
        min_index = mask.select(vp.first, min_index);
    }
    
    float thr = cos(angular_tolarance / 180 * float(M_PI));

    // cout << "incl = " << RowVectorXf(min_inclination) << endl;
    // cout << "idx = " << RowVectorXi(min_index) << endl;
    // cout << thr << endl;

    for (size_t i = 0; i < size_t(n_new_lines); ++i)
    {
        LineSegment & l = new_lines_array[i];
        if (min_inclination(i) > thr)
        {
            l.group_id = min_index(i);
        }
    }

}


void release_line_segments(LineSegment ** lines)
{
    if (*lines != NULL)
    {
        delete [] *lines;
        *lines = NULL;
    }
}


} // namespace
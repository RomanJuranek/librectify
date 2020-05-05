#include <iostream>
#include <list>

#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "config.h"
#include "liblgroup.h"
#include "image.h"
#include "line_detector.h"
#include "ransac.h"
#include "geometry.h"
#include "transform.h"


using namespace Eigen;
using namespace std;


namespace librectify {


static vector<LineSegment> _find_line_segment_on_buffer(InputPixelType * buffer, int width, int height, int stride)
{
    Image im = image_from_buffer(buffer, width, height, stride);
    // Detect the lines
    return find_line_segments(im, SEED_DIST, SEED_RATIO, TRACE_TOLERANCE);
}


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
    int * n_lines 
    )
{
    // Init image from buffer
    auto lines = _find_line_segment_on_buffer(buffer, width, height, stride);

    if (refine)
    {
        lines = postprocess_lines_segments(lines);
    }

    vector<LineSegment> filtered;
    filtered.reserve(lines.size());
    _filter_lines(lines.begin(), lines.end(), back_inserter(filtered), min_length);

    vector<LineSegment> groupped = group_lines(filtered);
    
    // Construct resulting array and copy data
    LineSegment * res = new LineSegment[groupped.size()];
    copy_n(groupped.begin(), groupped.size(), res);
    *n_lines = int(groupped.size());

    return res;
}


Point point_from_vector(const Vector3f & p)
{
    return Point{p.x(), p.y(), 1};
}

Vector3f vector_from_point(const Point & p)
{
    return Vector3f {p.x, p.y, p.z};
}

ImageTransform compute_rectification_transform_from_vp(
    int width, int height,
    const Point & vp_h, const Point & vp_v)
{
    auto tform = compute_image_transform(width, height, vector_from_point(vp_h), vector_from_point(vp_v));
    
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
    vector<VanishingPoint> vps = fit_vanishing_points(groupped);
    sort(vps.begin(), vps.end(), [](const VanishingPoint & a,const VanishingPoint & b){ return a.quality() > b.quality();});

    Vector3f image_center {float(width)/2, float(height)/2, 1};

    auto vp_v = select_vertical_point(vps, image_center, cfg.vertical_vp_angular_tolerance, height*cfg.vertical_vp_min_distance);
    auto vp_h = select_horizontal_point(vps, image_center, vp_v, width*cfg.horizontal_vp_min_distance);

    // vp_v.n_inliers > 0 means that the point was found

    Vector3f v1 = vp_h.coords;
    if (v1.z() != 0)
        v1.head(2) -= image_center.head(2);
    Vector3f v2 = vp_v.coords;
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
    ArrayXi groups = group_id(lines);
    MatrixX3f h = homogeneous(lines).rowwise().normalized();
    VectorXf wts = weigths(lines);
    VectorXf len = length(lines);

    // cout << h << endl << endl;
    // cout << wts << endl << endl;
    // cout << len << endl << endl;

    if (group < 0)
    {
        auto vp = VanishingPoint(h, len.cwiseProduct(wts));
        return point_from_vector(vp.coords);
    }
    else
    {
        auto idx = index_array(groups == group);
        auto vp = VanishingPoint(h(idx, all), len(idx).cwiseProduct(wts(idx)));
        return point_from_vector(vp.coords);
    }
}


void assign_to_group(
    const LineSegment * lines_array, int n_lines,
    LineSegment * new_lines_array, int n_new_lines,
    float angular_tolarance)
{
    vector<LineSegment> lines(lines_array, lines_array+n_lines);
    vector<VanishingPoint> vps = fit_vanishing_points(lines);

    vector<LineSegment> new_lines(new_lines_array, new_lines_array+n_new_lines);
    MatrixX2f a = anchor_point(new_lines);
    MatrixX2f d = direction_vector(new_lines).rowwise().normalized();
    
    ArrayXf min_inclination = ArrayXf::Zero(n_new_lines);
    ArrayXi min_index = ArrayXi::Zero(n_new_lines);

    // cout << "size: " << new_lines.size() << endl;
    // cout << "a=\n" << a << endl;
    // cout << "d=\n" << d << endl;

    for (size_t i = 0; i < vps.size(); ++i)
    {
        Vector3f p = vps[i].coords;
        ArrayXf x = inclination(a, d, p).array();

        // cout << "p: " << RowVector3f(p) << endl;
        // cout << "x: " << x.transpose() << endl;
        
        Array<bool,-1,1> mask = (x > min_inclination).eval();
        min_inclination = mask.select(x, min_inclination);
        min_index = mask.select(int(i), min_index);
    }
    
    float thr = cos(angular_tolarance / 180 * float(M_PI));

    // cout << "incl = " << RowVectorXf(min_inclination) << endl;
    // cout << "idx = " << RowVectorXi(min_index) << endl;
    // cout << thr << endl;

    for (size_t i = 0; i < n_new_lines; ++i)
    {
        LineSegment & l = new_lines_array[i];
        if (min_inclination(i) > thr)
        {
            l.group_id = min_index(i);
        }
    }

}


} // namespace
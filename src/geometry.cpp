#include <iostream>
#include <vector>
#include <list>

#include <Eigen/Dense>

#include "config.h"
#include "librectify.h"
#include "geometry.h"


using namespace std;
using namespace Eigen;


namespace librectify {


LineSegment fit_line_parameters(const Eigen::MatrixX2f & X, const Eigen::VectorXf & w)
{
    VectorXf weights = w / w.sum();  // Weights sum to 1

    Vector2f a = (X.array().colwise() * weights.array()).colwise().sum();
    MatrixX2f centered = X.rowwise() - a.transpose();
    
    // #if LGROUP_DEBUG_PRINTS
    // cout << "X=" << X << endl;
    // cout << "w=" << weights << endl;
    // cout << "a=" << a << endl;
    // cout << "centered=" << centered << endl;
    // cout << "centered.adjoint()=" << endl << centered.adjoint() << endl;
    // cout << "weights.diadonal()=" << endl << weights.diagonal() << endl;
    // #endif

    Matrix2f cov = centered.adjoint() * weights.asDiagonal() * centered;
    SelfAdjointEigenSolver<Matrix2f> eig(cov);
    //Matrix2f evec = eig.eigenvectors().colwise().normalized();

    Vector2f d = eig.eigenvectors().col(1);
    Vector2f n = eig.eigenvectors().col(0);

    VectorXf t = centered * d;
    float t0 = t.minCoeff();
    float t1 = t.maxCoeff();

    Vector2f c0 = a + d * t0;
    Vector2f c1 = a + d * t1;
        
    LineSegment l;
    l.x1 = c0(1);
    l.y1 = c0(0);
    l.x2 = c1(1);
    l.y2 = c1(0);
    l.weight = w.mean();
    l.err = (centered * n).cwiseAbs().mean();
    //l.err = eig.eigenvalues().coeff(0);
    l.group_id = -1; // Unassigned

    return l;
}


Vector3f homogeneous(const LineSegment & l)
{
    Vector3f a = {l.x1, l.y1, 1};
    Vector3f b = {l.x2, l.y2, 1};
    return a.cross(b);
}


Vector2f anchor_point(const LineSegment & l)
{
    return {(l.x2+l.x1) / 2, (l.y2+l.y1) / 2};
}


Vector2f direction_vector(const LineSegment & l)
{
    return {l.x2-l.x1, l.y2-l.y1};
}


Vector2f normal_vector(const LineSegment & l)
{
    return {-(l.y2-l.y1), l.x2-l.x1};
}


float length(const LineSegment & l)
{
    return direction_vector(l).norm();
}


Vector4f bounding_box(const vector<LineSegment> & lines)
{
    MatrixX2f coords(2*lines.size(), 2);
    for (size_t i = 0; i < lines.size(); ++i)
    {
        const LineSegment & l = lines[i];
        coords(2*i+0,all) = Vector2f(l.x1, l.y1);
        coords(2*i+1,all) = Vector2f(l.x2, l.y2);
    }
    Vector4f bb;
    bb.head<2>() = coords.colwise().minCoeff();
    bb.tail<2>() = coords.colwise().maxCoeff();
    return bb;
}


MatrixX3f homogeneous(const vector<LineSegment> & lines)
{
    MatrixX3f h(lines.size(), 3);
    for (size_t i = 0; i < lines.size(); ++i)
    {
        h.row(i) = homogeneous(lines[i]);
    }
    return h;
}


MatrixX2f anchor_point(const vector<LineSegment> & lines)
{
    MatrixX2f h(lines.size(), 2);
    for (size_t i = 0; i < lines.size(); ++i)
    {
        h.row(i) = anchor_point(lines[i]);
    }
    return h;
}


MatrixX2f direction_vector(const vector<LineSegment> & lines)
{
    MatrixX2f d(lines.size(), 2);
    for (size_t i = 0; i < lines.size(); ++i)
    {
        d.row(i) = direction_vector(lines[i]);
    }
    return d;
}


VectorXf reprojection_error(const vector<LineSegment> & lines)
{
    VectorXf err(lines.size());
    for (size_t i = 0; i < lines.size(); ++i)
    {
        err(i) = std::max(lines[i].err, 0.1f);
    }
    return err;
}


ArrayXi group_id(const vector<LineSegment> & lines)
{
    ArrayXi group(lines.size());
    for (size_t i = 0; i < lines.size(); ++i)
    {
        group(i) = lines[i].group_id;
    }
    return group;
}


VectorXf weigths(const vector<LineSegment> & lines)
{
    VectorXf w(lines.size());
    for (size_t i = 0; i < lines.size(); ++i)
    {
        w(i) = lines[i].weight;
    }
    return w;
}


static const Matrix2f M((Matrix2f() << 0, 1, -1, 0).finished());


MatrixX2f normal_vector(const vector<LineSegment> & lines)
{
    return direction_vector(lines) * M;
}


VectorXf length(const vector<LineSegment> & lines)
{
    return direction_vector(lines).rowwise().norm();
}


VectorXf inclination(const MatrixX2f & a, const MatrixX2f & d, const RowVector3f & p)
{
    MatrixX2f v;
    if (abs(p.z()) < EPS)
    {
        v = MatrixX2f::Zero(a.rows(),2);
        v.rowwise() = RowVector2f(p.x(), p.y());
    }
    else
    {
        RowVector2f p_norm = {p.x()/p.z(), p.y()/p.z()};
        v = (-a).rowwise() + p_norm;
    }
    v.rowwise().normalize();
   
    //cerr << ((v * d.transpose()).diagonal()).cwiseAbs().eval() << endl;
    return ((v * d.transpose()).diagonal()).cwiseAbs().eval();
}

} // namespace
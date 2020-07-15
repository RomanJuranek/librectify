#include <vector>
#include <iostream>

#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "config.h"
#include "cht.h"
#include "image.h"

using namespace Eigen;
using namespace std;

/*
Calculate line intersections with diagonals
*/
vector<MatrixX3f> line_intersections_diag(const MatrixX3f & l)
{
    int n = l.rows();
    auto a = l.col(0);
    //auto b = l.col(1);
    auto c = l.col(2);

    vector<MatrixX3f> D;
    
    MatrixX3f D_ST(n,3);
    D_ST.col(0).noalias() = a;
    D_ST.col(1).noalias() = c;
    D_ST.col(2).noalias() = a+c;
    D.push_back(D_ST);

    MatrixX3f D_SS(n,3);
    D_SS.col(0).noalias() = a;
    D_SS.col(1).noalias() = -c;
    D_SS.col(2).noalias() = a-c;
    D.push_back(D_SS);
    D.push_back(D_ST);
    D.push_back(D_SS);

    return D;
}


vector<MatrixX3f> line_intersections_axes(const MatrixX3f & l)
{
    int n = l.rows();
    auto a = l.col(0);
    auto b = l.col(1);
    auto c = l.col(2);

    vector<MatrixX3f> A;
    
    MatrixX3f A_ST(n,3);
    A_ST.col(0).noalias() = -b;
    A_ST.col(1).setConstant(0);
    A_ST.col(2).noalias() = -b+c;
    A.emplace_back(A_ST);

    MatrixX3f A_SS(n,3);
    A_SS.col(0).noalias() = b;
    A_SS.col(1).setConstant(0);
    A_SS.col(2).noalias() = b+c;
    A.emplace_back(A_SS);

    MatrixX3f A_TS(n,3);
    A_TS.col(0).setConstant(0);
    A_TS.col(1).noalias() = -b;
    A_TS.col(2).noalias() = a-b;
    A.emplace_back(A_TS);

    MatrixX3f A_TT(n,3);
    A_TT.col(0).setConstant(0);
    A_TT.col(1).noalias() = b;
    A_TT.col(2).noalias() = a+b;
    A.emplace_back(A_TT);

    return A;
}

float eps = 1e-5;


/*
Transform intersections with axes and diagonals to endpoints for accumulation
*/
MatrixX4f get_endpoints(const MatrixX3f & d, const MatrixX3f & x, const MatrixX3f & y)
{
    int n = d.rows();

    auto d_intersection = d.leftCols(2).array().colwise() / d.col(2).array();
    auto d_valid = (d_intersection.col(0).array() >= 0.f) &&
                   (d_intersection.col(0).array() <= 1.f) &&
                   (d_intersection.col(1).array() >= 0.f) &&
                   (d_intersection.col(1).array() <= 1.f) &&
                    d.col(2).array().abs() > eps;

    auto x_intersection = x.leftCols(2).array().colwise() / x.col(2).array();
    auto x_valid = (x_intersection.col(0).array() >= 0) &&
                   (x_intersection.col(0).array() <= 1) &&
                   //(x_intersection.col(1).array() >= 0) &&
                   //(x_intersection.col(1).array() <= 1) &&
                    x.col(2).array().abs() > eps;

    auto y_intersection = y.leftCols(2).array().colwise() / y.col(2).array();
    auto y_valid = //(y_intersection.col(0).array() >= 0) &&
                   //(y_intersection.col(0).array() <= 1) &&
                   (y_intersection.col(1).array() >= 0) &&
                   (y_intersection.col(1).array() <= 1) &&
                    y.col(2).array().abs() > eps;

    
    auto v = ((d_valid || x_valid || y_valid) == 1).eval();

    
    // cout << "D\n" << d_intersection << endl << "valid\n" << d_valid << endl << endl;
    // cout << "X\n" << x_intersection << endl << "valid\n" << x_valid << endl << endl;
    // cout << "Y\n" << y_intersection << endl << "valid\n" << y_valid << endl << endl;
    // cout << "v:\n" << v << endl << "count: " << v.count() << endl;


    MatrixX4f endpoints((v==1).count(), 4);
    int row = 0;
    for (int i = 0; i < n; ++i)
    {
        if (v(i) == 1)
        {
            //cout << "i=" << i << endl;
            int c = 0;
            if (d_valid(i))
            {
                endpoints.row(row).segment(c,2) = d_intersection.row(i);
                c += 2;
            }
            if (x_valid(i))
            {
                endpoints.row(row).segment(c,2) = x_intersection.row(i);
                c += 2;
            }
            if (y_valid(i) && c < 4)
            {
                endpoints.row(row).segment(c,2) = y_intersection.row(i);
                c += 2;
            }
            // cout << "c=" << c << endl;
            // if (c < 4)
            // {
            //     cerr << "This should not happen!!!" << endl;
            // }
            row++;
        }
    }

    return endpoints;
}

template<typename Derived>
void accumulate_lines(
    ArrayBase<Derived> & acc,
    const ArrayX4f & endpoints)
{
    int n = endpoints.rows();

    auto d = endpoints.leftCols(2) - endpoints.rightCols(2);
    auto dx = d.col(0);
    auto dy = d.col(1);
    auto dir_flag = dx.abs() > dy.abs();

    auto steps = (dx.abs().max(dy.abs())).round() + 1;

    auto x0 = endpoints.col(0).round();
    auto y0 = endpoints.col(1).round();
    auto x1 = endpoints.col(2).round();
    auto y1 = endpoints.col(3).round();

    for (int i = 0; i < n; ++i)
    {
        int num_steps = steps(i);
        ArrayX2f px(num_steps, 2);
        px.col(0) = ArrayXf::LinSpaced(num_steps, x0(i), x1(i));
        px.col(1) = ArrayXf::LinSpaced(num_steps, y0(i), y1(i));
        ArrayX2i px_int = px.round().cast<int>();

        //cout << px_int << endl << endl;

        for (int j = 0; j < num_steps; ++j)
        {
            acc(px_int(j,1), px_int(j,0)) += 1;
        }
    }
}



DiamondSpace::DiamondSpace(int sz)
    :SS(SubspaceAccumulator(sz, 1, 1))
{
}

DiamondSpace::~DiamondSpace()
{
}

void DiamondSpace::clear()
{
    SS.acc.setZero();
}

void DiamondSpace::accumulate(const Lines & lines, const Eigen::VectorXf & weights)
{   
    auto D = line_intersections_diag(lines);
    auto A = line_intersections_axes(lines);

    Segments seg_ST = get_endpoints(D[0], A[0], A[3]);
    cerr << "segs=\n" << (seg_ST.array() * (size-1)).round().cast<int>() << endl;

    accumulate_lines(ST, seg_ST * (size-1));

    Segments seg_SS = get_endpoints(D[1], A[1], A[3]);
    accumulate_lines(SS, seg_SS * (size-1));

    Segments seg_TT = get_endpoints(D[2], A[1], A[2]);
    accumulate_lines(TT, seg_TT * (size-1));

    Segments seg_TS = get_endpoints(D[3], A[0], A[2]);
    accumulate_lines(TS, seg_TS * (size-1));
}

float DiamondSpace::argmax(Point & p) const
{
    Index r, c;
    float max_val = acc.maxCoeff(&r, &c);
    p = Point(c, r, 1);
    return max_val;
}


DiamondSpace lines_to_cht(vector<LineSegment> & lines, int resolution)
{

}

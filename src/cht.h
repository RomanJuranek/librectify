/*

TODO
----
* correct orientation of intersections so that the pointe are mapped
  in the same orientation as in the original space - i.e. just shift, no need to set origin point to (0,0)
  This will enable to map the spaces to the one big accumulator seamlesly.
  * line_intersection_axes/diag
* Check mapping from pixel index to DS coordinate
* backward mapping from accumulator index u,v to x,y in cartesian space


How it should work
------------------
DiamondSpace works with lines which is MatrixX3f with (a,b,c) homogenous
coordinates. The lines must be normalized so that they lie in reasonable range
around 0.

accumulate() will insert multiple weighted lines to the accumulator. Note that
the weights can be negative (so lines can be removed!).

argmax() will identify point (in the space of normalized lines) where many lines
intersect. This point must be de-normalized to obtain the point in the original
space.


Example
-------

We have endpoints of line segments in image of size W,H (i.e. pixel coordinates)

    MatrixX4f endpts;

Extract homogenous coordinates of endpoint to pt_a and pt_b

    MatrixX3f pt_a, pt_b;
    pt_a.leftCols(2) = endpts.leftCols(2)
    pt_a.col(3) = 1;
    pt_b.leftCols(2) = endpts.rightCols(2)
    pt_b.col(3) = 1;

Normalize the endpoints and form the lines as a cross product. And again
normalize so lines are unit vectors.

    Normalizer N(max(H,W), {W/2,H/2});
    pt_a = N.forward(pt_a);
    pt_b = N.forward(pt_b);
    lines = pt_a.cross(pt_b).normalize();
    weights = VectorXf::Ones(lines.rows());

Create accumulator and insert lines.

    DiamondSpace D(128);
    D.accumulate(lines, weights);

Identify maximum point and de-normalize to original image space.

    Vector3f p;
    float max_val = D.argmax(p);
    p = N.backward(p);

Now p is the point where the lines in image intersect.

The process should be transparent for the users.


    D.accumulate(InputIterator first, InputIterator last, float weights scale);
    D.argmax()


*/

#pragma once


#include <iostream>
#include <Eigen/Core>


namespace librectify {

namespace cht {


class DiamondAccumulator
{
private:
    using AccumulatorType = Eigen::ArrayXXf;

    int d; // size of space
    AccumulatorType SS, ST, TS, TT;

    int n_lines {0};
    
public:
    DiamondAccumulator(int _d)
        :d(_d)
    {
        SS = AccumulatorType(d,d);
        // ...
        clear();
    }
    ~DiamondAccumulator();

    void clear()
    {
        SS.setZero(); 
        // ...
    }

    // Add lines to to accumulator
    void insert(const Eigen::MatrixX3f & h, const Eigen::VectorXf & w);

    float argmax(Eigen::Vector3f & p) const;

    // Mapping R2 <-> DS
    Eigen::MatrixX3f transform(const Eigen::MatrixX3f & x) const;
    Eigen::MatrixX3f inverse(const Eigen::MatrixX3f & x) const;
};

} // namespace cht

} // namespace librectify
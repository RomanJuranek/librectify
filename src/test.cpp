#include <iostream>

#include "librectify.h"

using namespace std;
using namespace librectify;


ostream & operator<<(ostream & out, const Point & p)
{
    out << "[" << p.x << "," << p.y << "," << p.z << "]";
    return out;
}


int main()
{

    LineSegment A = {0, 0, 10,0, 1,0, 10};
    LineSegment B = {10,0, 8,5, 1,0, 1};
    LineSegment C = {8,5, 2,5, 1,0, 10};
    LineSegment D = {2,5, 0,0, 1,0, 1};

    LineSegment ls[4] = {A,B,C,D};
    int n_lines = 4;

    Point vp1 = fit_vanishing_point(ls, n_lines, 1);
    cout << "vp1=" << vp1 << endl;

    Point vp2 = fit_vanishing_point(ls, n_lines, 10);
    cout << "vp2=" << vp2 << endl;

    Point vp3 = fit_vanishing_point(ls, n_lines, -1);
    cout << "vp3=" << vp3 << endl;


    LineSegment line = {5,1, 5,4, 1,0, -1};

    assign_to_group(ls, n_lines, &line, 1, 10);

    cout << "Assigned to group "<< line.group_id << endl;

    return 0;
}
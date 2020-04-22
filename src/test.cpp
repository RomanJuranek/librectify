#include <iostream>

#include "liblgroup.h"

using namespace std;

ostream & operator<<(ostream & out, const Point & p)
{
    out << "[" << p.x() << "," << p.y() << "," << p.z() << "]";
    return out;
}


int main()
{
    LineSegment ls[4];
    ls[2] = {0,-1, 1,0, 1,0, 10}; 
    ls[3] = {0, 1, 1,0, 1,0, 10};

    ls[0] = {0,-1,-1,0, 1,0, 0};
    ls[1] = {0, 1,-1,0, 1,0, 0};

    int n_lines = 4;

    Point vp1 = fit_vanishing_point(ls, n_lines, 0);
    cout << "vp1=" << vp1 << endl;

    Point vp2 = fit_vanishing_point(ls, n_lines, 1);
    cout << "vp2=" << vp2 << endl;

    Point vp3 = fit_vanishing_point(ls, n_lines, -1);
    cout << "vp3=" << vp3 << endl;


    LineSegment line = {-1,-1, 1, 0, 1,0, -1};

    find_closest_group(ls, n_lines, &line, 1, 10);

    cout << "Assigned to group "<< line.group_id << endl;

    return 0;
}
/*
Public API for line segment detection, groupping and vanishing point estimation

(c) 2020, Roman Juranek <ijuranek@fit.vutbr.cz>

Development of this software was funded by TACR project TH04010394, Progressive
Image Processing Algorithms.

Data model
----------
We work with GRAYSCALE image in float array. We expect a pointer to a buffer,
image dimensions (width and height), and stride between rows (which can be
negative).

Image of size W,H with stride S at address X can be passed either as 1/
buffer=X, stride=S, or 2/ buffer=X+S*(H-1), stride=-S

Both versions are semantically equivalent.

*/


#pragma once


#ifdef _WINDOWS
    #ifdef LIBLGROUP_EXPORTS
        #define DLL_PUBLIC __declspec(dllexport)
    #else
        #define DLL_PUBLIC __declspec(dllimport)
    #endif
#else
    #define DLL_PUBLIC
#endif


// namespace lgroup
// {

/*
Structure which holds a single line segment and its properties.
*/
struct LineSegment
{
    // Coordinates of endpoints
    float x1, y1, x2, y2;
    // Line weight - relative importance in in vanishing point calculation
    float weight;
    // Reprojection error - straight and thin lines have err close to zero
    float err;
    // Group to which the line belongs to (-1 means no group was assigned)
    int group_id;
};


/*
Structure which holds spatial points in homogenous coordinates.
*/
struct Point
{
    float data[3];
    Point():
        data{0,0,0} {}
    Point(float x, float y, float z):
        data{x,y,z} {}
    float & operator[](int i) {return data[i]; }
    float operator[](int i) const {return data[i]; }
    float x() const { return data[0]; }
    float & x() { return data[0]; }
    float y() const { return data[1]; }
    float & y() { return data[1]; }
    float z() const { return data[2]; }
    float & z() { return data[2]; }
    bool is_finite() const { return z() < 1e-6; }
};

/*
Definition of projective image transform - i

Suppose an image of size (W,H)
Its corners are : [0,0], [W-1,0], [0,H-1], and [W-1,H-1]

The structure specifies where the corners of an image moves.

[0,0] -> top_left
etc.

From this quadrangle, projective transform can be computed
and the image can be transformed accordingly.
*/
struct ImageTransform
{
    int width;
    int height;
    Point top_left, top_right, bottom_left, bottom_right;
    Point horizontal_vp;
    Point vertical_vp;
};


/*
Set number of threads to be used by the library.
The function sets internal state and does not
interfere with global OpenMP settings.
*/
extern "C" void DLL_PUBLIC set_num_threads(int t);

extern "C" int DLL_PUBLIC get_num_threads();


/*
Detect line segments in image.

Inputs
------
buffer : Pointer to image buffer. Layout is described above.
width, height : Image dimensions in pixels
stride : Step between rows in array elements (not bytes!)
n_lines : Ptr to int where the number of detected lines will be returned
min_length : Only lines longer than min_length will be returned

Output
------
Pointer to allocated array of LineSegment structures. The array can be safely
deleted when no longer required.
 */

using InputPixelType = float;  // Do not change this!!!

extern "C" DLL_PUBLIC LineSegment * find_line_segment_groups(
    InputPixelType * buffer, int width, int height, int stride,
    float min_length,
    int * n_lines);


enum RectificationStrategy
{
    ROTATE_H,
    ROTATE_V,
    RECTIFY,
    KEEP,
};

/*
Parameters for image rectification
*/
struct RectificationConfig
{
    // Search space for vertical vanishing point (degrees)
    float vertical_vp_angular_tolerance {40};
    // Minimal distance of vertical VP from from image center (ratio of image height)
    float vertical_vp_min_distance {0.7};
    // Strategy for vertical direction processing
    RectificationStrategy v_strategy {RECTIFY};

    // Minimal distance of horizontal VP from from image center (ratio of image width)
    float horizontal_vp_min_distance {0.7};
    // Strategy for horizontal direction processing
    RectificationStrategy h_strategy {RECTIFY};
    // force_half_plane
};

/*
Compute rectification from line groups

The function automatically identifies horizontal and vertical vanishing point by angular
difference from horizonral, vertical direction. If none is found withinn +- 30 degrees,
ideal point in infinity ([1,0,0] for horizontal and [0,1,0] for vertical) is used.
From these points homography transformation is computed and coordinates of transformed
image corners are returned.

Inputs
------
lines : Array with lines as returned from find_line_segment_groups
n_lines : number of lines in array

Notes
-----
The returned coordinates are in the coordinate system where image center is in does not move.
To calculate the final coordinates for transformation is application specific process
involving clipping (in case the resulting image would be too large) and translation
to image coordinates. These steps are not part of the library and must be
implemented by the user.

*/
extern "C" DLL_PUBLIC ImageTransform compute_rectification_transform(
    LineSegment * lines, int n_lines,
    int width, int height,
    const RectificationConfig & cfg);


extern "C" DLL_PUBLIC ImageTransform compute_rectification_transform_from_vp(
    int width, int height,
    const Point & vp_h, const Point & vp_v);


/*
Fit vanishing point to a line group.

Inputs
------
lines : Array with lines as returned from find_line_segment_groups
n_lines : number of lines in array
group : id of group to be used for fitting. if group<0, the function fits the point to ALL the lines.

Output
------
Coordinates of vanishing point in image space
*/
extern "C" DLL_PUBLIC Point fit_vanishing_point(
    const LineSegment * lines, int n_lines,
    int group);

/*
Find closest group for new lines

Inputs
------
lines_array, n_lines: Lines with assigned group_id
new_lines, n_new_lines: Lines to which group_id will be asigned
angular_tolerance : tolerance in degrees.

The function takes vanishing points of lines in lines_array and
assigns each line in new_lines to the closes group based on
the angular inclination between vanishing points and the line.
Lines outide angular_tolerance are kept unchanged.

WARNING: new_lines are modified (group_id member)
*/    
extern "C" DLL_PUBLIC void assign_to_group(
    const LineSegment * lines_array, int n_lines,
    LineSegment * new_lines_array, int n_new_lines,
    float angular_tolarance);


// } // namespace lgroup

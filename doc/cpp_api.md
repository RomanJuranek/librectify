# API documentation

The library API is enclosed in the namespace `librectify` and it procides basci functions for analysis of lines in image and calculating image transforms.

## Data model

We work with **grayscale** image in `float` array. We expect a pointer to a buffer, image dimensions (width and height), and stride between rows (which can be negative).

Image of size `WxH` with stride `S` at address `X` can be passed either as
1. buffer=X, stride=S, or
2. buffer=X+S*(H-1), stride=-S

Both versions are semantically equivalent.

## Types

### `InputPixelType`

```c++
using InputPixelType = float
```

Expected data type of image pixels.

## Enums

### `RectificationStrategy`

```c++
enum RectificationStrategy
{
    ROTATE_H,
    ROTATE_V,
    RECTIFY,
    KEEP,
};
```

Controls how the transform is calculated in `RectificationConfig`. This can be set for vertical and horizontal direction independently. The meaning of the individual values is:

* `ROTATE_H` - Rotation by angle defined by horizontal vanishing point.
* `ROTATE_V` - Rotation by angle defined by vertical vanishing point.
* `RECTIFY` - Full rectification - alignment of lines with an axis
* `KEEP` - Do nothing

## Structures

### `LineSegment`
```c++
struct LineSegment
{
    float x1, y1, x2, y2;
    float weight;
    float err;
    int group_id;
};
```

Holds a single line segment with endpoints `(x1,y1)` and `(x2,y2)`. `weights` is an average response of edge detector on pixels belonging to the segment, and `err` is average reprojection error of the pixles. `group_id` is the identifier of the group to which the line was assigned - lines with the same `group_id` points towards common vanishing point.

### `Point`

```c++
struct Point
{
    float x, y, z;
};
```

A point at location `(x,y)`. It can be either finite (`z==1`) or infinite (`z==0`) in which case `(x,y)` is direction in which the point lies.

### `ImageTransform`

```c++
struct ImageTransform
{
    int width;
    int height;
    Point top_left, top_right, bottom_left, bottom_right;
    Point horizontal_vp;
    Point vertical_vp;
};
```

Definition of image transformation. `width` and `height` of is the image size in pixels. `top_left`... fields are new coordinates of the image corners and `horizontal_vp` and `vertical_vp` are coordinates of vanishing points from which the transform was calculated. Note that locations of vanishing points are in the image space of the source image. New coordinates are in different space and must be translated to correct locations (which is intentionaly left for the user).

![xx](transform.svg)


### `RectificationConfig`

```c++
struct RectificationConfig
{
    float vertical_vp_angular_tolerance;
    float vertical_vp_min_distance;
    RectificationStrategy v_strategy;
    float horizontal_vp_min_distance;
    RectificationStrategy h_strategy;
};
```

Configures how the transformation will be computed.

Vertical vanishing point will be searched in vertical direction in `(-a, +a)` degrees (where `a` is `vertical_vp_angular_tolerance`) at minimum distance `vertical_vp_min_distance`.

Horizontal vansihinf point will be searched in lower hald-plane at minimum distance `horizontal_vp_min_distance`.

These search space constraints are important for

## Interface

### `set_num_threads`

```c++
void set_num_threads(int t);
```

Sets the number of threads used by the library. This setting does not affect OpenMP number of threads. Internally, it only sets a variable used by parallel regions. Setting `t` to negative value (or to `1`) disables threading altogether. Setting to `0` forces global OpenMP number of threads (as set by `omp_set_num_threads`). Setting any positive value sets the number of threads (capped by `omp_max_threads`).

### `get_num_threads`

```c++
int get_num_threads();
```

Returns the internal number of threads. Default value is `0`.

### `find_line_segment_groups`

```c++
LineSegment * find_line_segment_groups(InputPixelType * buffer, int width, int height, int stride, float min_length, bool refine, int * n_lines);
```

### `release_line_segments`
```c++
void release_line_segments(LineSegment * lines);
```

### `compute_rectification_transform`
```c++
ImageTransform compute_rectification_transform(LineSegment * lines, int n_lines, int width, int height, const RectificationConfig & cfg);
```

### `compute_rectification_transform_from_vp`
```c++
ImageTransform compute_rectification_transform_from_vp(
int width, int height, const Point & vp_h, const Point & vp_v);
```
### `fit_vanishing_point`
```c++
Point fit_vanishing_point(const LineSegment * lines, int n_lines, int group);
```

### `assign_to_group`
```c++
void assign_to_group(const LineSegment * lines_array, int n_lines, LineSegment * new_lines_array, int n_new_lines, float angular_tolarance);
```

---
&copy; 2020 Roman Juranek
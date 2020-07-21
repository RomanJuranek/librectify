### Whats new in **v0.9.9**
* VP estmation completely rewritten - besides RANSAC, PROSAC or Hough transform can be used

### Whats new in **v0.9.8**
* Removed `set_num_threads`/`get_num_threads`. Threading is controlled by `num_threads` parameter in `find_line_segment_groups`.

### Whats new in **v0.9.7**
* Thread safety in RANSAC improved
* `release_line_segments` takes ptr to `LineSegment*` and sets the pointer to 0 after delete
* Changes of some internal representations and small performance improvements
* Fixed error in `compute_rectification_transform_from_vp`

### Whats new in **v0.9.6**
* Fixed crash in line refinement
* Fixed vanishing points in `ImageTransform`
* Improved compatibility with Visual Studio (mostly warnings)
* Default behaviour of autorectify changed to full rectification
* Autorectify now uses automatic bounding box for transformed image
* Documentation

### Whats new in **v0.9.5**
* Experimental support for line postprocessing (parameter `refine` in `find_line_segment_groups`)
* Fixed threading issues
  * Some parallel regions did not use internal number of threads
  * By default, omp setting is used
* API is enclosed in namespace `librectify`
* `release_line_segments` for deallocation of memory retuned by `find_line_segment_groups`

### Whats new in **v0.9.4**
* `set_num_threads` sets internal number of threads and does not interfere with global OpenMP settings.
* `get_num_threads` get number of threads used by the library
* `fit_vanishing_point` function gets the point for a single group
* `find_closest_group` assigns lines to groups
* Improved logic for selection of vanishing points and transform computation
* `RectificationConfig` specifies how each direction is rectified with `RectificationStrategy` (allows for 15 different transforms including pure rotations)
* Fixed numerical instability in transform computation.
* `ImageTransform` now contains the actual vanishing points used for transform computation.
* Command line arguments in `autorectify` for selection of the transform
* Fixed error in line filtering (which caused that the filter had no effect)
* More compact lines, and lower number of lines - improves stability
* `Point` is pure struct with `x`, `y`, `z` members with no c++ interface

### Whats new in **v0.9.3**
* Support for negative stride
* Buffer format back to `float*`
* Speed improvements in line group estimation and line detection
* Parameter tuning to improve results
* API changes to simplify usage - see `liblgroup.h`
* Added `set_num_threads` - multithreading via OpenMP
* Image transform computation based on automatic identification of vanishing points
* `autorectify.cpp` application (replaces test.cpp)
* Doc update
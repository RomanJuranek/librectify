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
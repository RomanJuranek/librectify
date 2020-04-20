#pragma once 

#include <vector>

#include "image.h"
#include "liblgroup.h"


std::vector<LineSegment> find_line_segments(const Image & image, int seed_dist, float seed_ratio, float mag_tolerance);

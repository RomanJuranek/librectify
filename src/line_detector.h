#pragma once 

#include <vector>

#include "image.h"
#include "librectify.h"

namespace librectify {

std::vector<LineSegment> find_line_segments(const Image & image, int seed_dist, float seed_ratio, float mag_tolerance);

std::vector<LineSegment> postprocess_lines_segments(const std::vector<LineSegment> & lines);

} // namespace
#pragma once 

#include <vector>

#include "image.h"
#include "librectify.h"
#include "threading.h"

namespace librectify {

std::vector<LineSegment> find_line_segments(const Image & image, int seed_dist, float seed_ratio, float mag_tolerance, const ThreadContext & ctx);

std::vector<LineSegment> postprocess_lines_segments(const std::vector<LineSegment> & lines, const ThreadContext & ctx);

} // namespace
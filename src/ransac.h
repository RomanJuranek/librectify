
#pragma once

#include <list>
#include <vector>

#include "librectify.h"

namespace librectify {

std::vector<LineSegment> group_lines(std::vector<LineSegment> & lines, const ThreadContext & ctx);

} // namespace
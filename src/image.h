#pragma once

#include <Eigen/Core>
#include "config.h"

namespace librectify {

using Image = Eigen::Array<InputPixelType,-1,-1, Eigen::RowMajor>;

using Image_int = Eigen::Array<int,-1,-1, Eigen::RowMajor>;

using Mask = Eigen::Array<bool,-1,-1, Eigen::RowMajor>;

Image image_from_buffer(InputPixelType * buffer, int width, int height, int stride);

} //namespace
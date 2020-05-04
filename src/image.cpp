#include <Eigen/Core>
#include "config.h"
#include "image.h"


using namespace Eigen;

namespace librectify {

Image image_from_buffer(InputPixelType * buffer, int width, int height, int stride)
{
    using MapType = Eigen::Map<Image,Eigen::RowMajor,Eigen::Stride<-1,1>>;
    if (stride < 0)
    {
        buffer = buffer + (height-1)*stride;
        stride = -stride;
    }
    return MapType(buffer,height,width,Eigen::Stride<-1,1>(stride,1));
}

} // namespace
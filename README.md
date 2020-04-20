# `librectify` v0.9.4
_Apr 20, 2020_

&copy; 2020, Roman Juranek <ijuranek@fit.vutbr.cz>

Minimalistic library for image perspective correction. The library prvides API for automated identification of converging line groups and image transform computation from automatic identification of two orthogonal vanishing points. The library does not warp the image, this is in the hands of the user.

_Development of this software was funded by
TACR project TH04010394, Progressive Image Processing Algorithms._

## Dependencies
The library requires no external dependencies (apart from standard library and OpenMP runtime). Internally we use Eigen for representation of images and linear algebra.
Example app uses OpenCV for image IO and warping.

---

[Change log](ChangeLog) | [TODO](TODO)
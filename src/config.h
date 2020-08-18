/*
*/

#pragma once

// When set to 0, output stays clear
#define LGROUP_DEBUG_PRINTS 0


#define EPS 1e-6f

///////////////////////////////////////////////////////////////////////////////
// Input

// Data type of input buffer
using InputPixelType = float;

///////////////////////////////////////////////////////////////////////////////
// LINE GROUPS

// // The maximal number of groups to detect. Usually 2 to 4 groups have useful
// // geometric meaning.
#define MAX_MODELS 4

#define ESTIMATOR_INLIER_MAX_ANGLE_DEG 2.0f

#define ESTIMATOR_GARBAGE_MAX_ANGLE_DEG 4.0f

#define RANSAC_MAX_ITER 10000

///////////////////////////////////////////////////////////////////////////////
// Line detection

// Radius of edge detection kernel. The size is 2*EDGE_KERNEL_SIZE+1
#define EDGE_KERNEL_SIZE 2

// Sigma of gaussian derivative for edge detection
#define EDGE_KERNEL_SIGMA 1.0f

// Minimal distance of seed points for line tracing
#define SEED_DIST 2

// The ration of seed points to consider for tracing
#define SEED_RATIO 0.95

// Magnitude tolerance during tracing
#define TRACE_TOLERANCE 0.25f

// Mean reprojection error of pixels on a line
#define LINE_MAX_ERR 2.0f

// Mean weight of pixels on a line
#define LINE_MIN_WEIGHT 0.05f

// Minimal line length
#define LINE_MIN_LENGTH 5.f

// Minimal number of pixel of image component for line fitting
#define COMPONENT_MIN_SIZE 5

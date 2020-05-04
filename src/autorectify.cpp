/*
Test application showing basic principles of liblgroup library

(c) 2020, Roman Juranek <ijuranek@fit.vutbr.cz>

Development of this software was funded by TACR project TH04010394, Progressive
Image Processing Algorithms.
*/


#include <iostream>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iterator>

#include "liblgroup.h"

#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;


// Write range of lines to output
template<class II>
void dump_lines_csv(II first, II last, ostream & f)
{
    while(first != last)
    {
        const LineSegment & l = *first;
        f << l.x1 << "," << l.y1 << ","
          << l.x2 << "," << l.y2 << ","
          << l.weight << "," << l.err << ","
          << l.group_id << endl;
        ++first;
    }
}

void dump_tform(const ImageTransform & t, ostream & f)
{
    f << t.top_left.x << "," << t.top_left.y << endl;
    f << t.top_right.x << "," << t.top_right.y << endl;
    f << t.bottom_left.x << "," << t.bottom_left.y << endl;
    f << t.bottom_right.x << "," << t.bottom_right.y << endl;
}

// Downscale image if required (larger than max_size) and return scale
Mat prescale_image(Mat & im, int max_size, float & scale)
{
    int w = max(im.cols, im.rows);
    scale = min(float(max_size) / w, 1.0f);
    if (scale == 1.0)
    {
        return im;
    }
    Mat resized;
    resize(im, resized, Size(), scale, scale, INTER_AREA);
    return resized;
}


// Draw range of lines to the image
template <class II>
void draw_lines(II first, II last, Mat & image)
{
    vector<Scalar> colors;
    colors.push_back( Scalar(0,0,255) );
    colors.push_back( Scalar(0,255,0) );
    colors.push_back( Scalar(255,0,0) );
    colors.push_back( Scalar(0,255,255) );
    colors.push_back( Scalar(255,0,255) );
    colors.push_back( Scalar(255,255,0) );

    colors.push_back( Scalar(0,0,128) );
    colors.push_back( Scalar(0,128,0) );
    colors.push_back( Scalar(128,0,0) );
    colors.push_back( Scalar(0,128,128) );
    colors.push_back( Scalar(128,0,128) );
    colors.push_back( Scalar(128,128,0) );

    while(first != last)
    {
        const LineSegment & l = *first;
        int g = l.group_id;
        Scalar clr;
        int w;
        if (g == -1)
        {
            clr = Scalar(255,255,255);
            w = 1;
        }
        else
        {
            clr = colors[(g)%12];
            w = 3;
        }
        line(image, cv::Point(l.x1, l.y1), cv::Point(l.x2, l.y2), clr, w);
        circle(image, cv::Point(l.x1, l.y1), 7, clr, -1);
        circle(image, cv::Point(l.x2, l.y2), 7, clr, -1);
        ++first;
    }
}

// Wrapper for find_line_segment_groups accepting OpenCV images of any size
LineSegment * detect_line_groups(const Mat & image, int * n_lines)
{
    Mat image_f = image;
    image.convertTo(image_f, CV_32F, 1/256.0);
    
    float scale;
    image_f = prescale_image(image_f, 1200, scale);

    float * buffer = (float*)image_f.data;
    int w = image_f.cols;
    int h = image_f.rows;
    int stride = w;
    int min_length = float(max(h,w)) / 100.0f;

    LineSegment * lines = find_line_segment_groups(buffer, w, h, stride, min_length, true, n_lines);

    // Scale lines back to the original image
    for (size_t i = 0; i < *n_lines; ++i)
    {
        lines[i].x1 /= scale;
        lines[i].y1 /= scale;
        lines[i].x2 /= scale;
        lines[i].y2 /= scale;
    }

    return lines;
}


// Calculate homography matrix from known corner locations
Mat homography_from_corners(const ImageTransform & t)
{
    vector<Point2f> src;
    src.push_back(Point2f(0,       0));
    src.push_back(Point2f(t.width, 0));
    src.push_back(Point2f(0,       t.height));
    src.push_back(Point2f(t.width, t.height));

    vector<Point2f> dst;
    dst.push_back(Point2f(t.top_left.x,    t.top_left.y));
    dst.push_back(Point2f(t.top_right.x,   t.top_right.y));
    dst.push_back(Point2f(t.bottom_left.x, t.bottom_left.y));
    dst.push_back(Point2f(t.bottom_right.x,t.bottom_right.y));

    Mat H = findHomography(src, dst);
    return H;
}


// Split string with delimiter and return vector of its parts
std::vector<std::string> splitpath(const std::string& str, const std::set<char> delimiters)
{
    std::vector<std::string> result;
    char const* pch = str.c_str();
    char const* start = pch;
    for(; *pch; ++pch)
    {
        if (delimiters.find(*pch) != delimiters.end())
        {
            if (start != pch)
            {
                std::string str(start, pch);
                result.push_back(str);
            }
            else
            {
                result.push_back("");
            }
            start = pch + 1;
        }
    }
    result.push_back(start);
    return result;
}


// Join vector of string with delimiter
template <class II>
string join(II first, II last, const char* delim)
{
    stringstream res;
    copy(first, last, ostream_iterator<string>(res, delim));
    return res.str();
}

RectificationStrategy strategy_from_string(string s)
{
    if (s == "rotate_v")
    {
        return ROTATE_V;
    }
    else if (s == "rotate_h")
    {
        return ROTATE_H;
    }
    else if (s == "rectify")
    {
        return RECTIFY;
    }
    else
    {
        return KEEP;
    }
}

struct Options
{
    RectificationStrategy h_strategy {KEEP};
    RectificationStrategy v_strategy {KEEP};
    string filename {""};
    string suffix {"warp"};
};

template <class II>
Options process_arguments(II first, II last)
{
    Options opt;
    while (first != last)
    {
        if (*first == "-h")
        {
            first++;
            opt.h_strategy = strategy_from_string(*first);
        }
        else if (*first == "-v")
        {
            ++first;
            opt.v_strategy = strategy_from_string(*first);
        }
        else if (*first == "-s")
        {
            ++first;
            opt.suffix = *first;
        }
        else
        {
            opt.filename = *first;
        }
        ++first;
    }
    return opt;
}


int main(int argc, char ** argv)
{
    list<string> args(argv+1, argv+argc);
    Options opts = process_arguments(args.begin(), args.end());
 
    if (opts.filename == "")
    {
        cerr << "No input file" << endl;
        exit(1);
    }

    auto file_parts = splitpath(opts.filename, {'/'});
    string filename = file_parts[file_parts.size()-1];

    Mat image_rgb = imread(opts.filename, IMREAD_COLOR);
    if (!image_rgb.data)
    {
        cerr << "Error: Could not load " << opts.filename << endl;
        exit(1);
    }

    Mat image_8uc;
    cvtColor(image_rgb, image_8uc, COLOR_BGR2GRAY);

    ///////////////////////////////////////////////////////////////////////////

    // Set number of threads for parallelization
    set_num_threads(4);

    // Detect lines in image
    int n_lines = 0;
    LineSegment * lines = detect_line_groups(image_8uc, &n_lines);
    // Now we have n_lines segments lines[0] .. lines[n_lines-1]
    // We can modify them, e.g. add user defined lines (remember to assign them to the correct group)

    // Setup parameters for rectification
    RectificationConfig cfg;
    cfg.h_strategy = opts.h_strategy;
    cfg.v_strategy = opts.v_strategy;

    // Get the locations of corners
    ImageTransform t = compute_rectification_transform(lines, n_lines, image_8uc.cols, image_8uc.rows, cfg);

    // Calculate transform for image warping
    Mat H = homography_from_corners(t);

    ///////////////////////////////////////////////////////////////////////////

    Mat image_warped;
    warpPerspective(image_rgb, image_warped, H, Size(image_rgb.cols, image_rgb.rows));

    Mat image_gray;
    cvtColor(image_8uc, image_gray, COLOR_GRAY2BGR);
    draw_lines(lines, lines+n_lines, image_gray);

    string dst_dir = join(file_parts.begin(), file_parts.end()-1, "/");

    imwrite(dst_dir+filename+"_"+opts.suffix+".jpg", image_warped);
    imwrite(dst_dir+filename+"_"+opts.suffix+"_lines.jpg", image_gray);
    ofstream line_cache(dst_dir+filename+"_"+opts.suffix+"_lines.csv");
    dump_lines_csv(lines, lines+n_lines, line_cache);
    line_cache.close();
    ofstream tform_cache(dst_dir+filename+"_"+opts.suffix+"_tform.csv");
    dump_tform(t, tform_cache);
    tform_cache.close();

    delete [] lines; // cleanup
}

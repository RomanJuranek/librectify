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

#include "librectify.h"

#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;
using namespace librectify;


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
    f << t.horizontal_vp.x << "," << t.horizontal_vp.y << "," << t.horizontal_vp.z << endl;
    f << t.vertical_vp.x << "," << t.vertical_vp.y << "," << t.vertical_vp.z << endl;
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
    colors.push_back( Scalar(0,0,255) ); // r
    colors.push_back( Scalar(0,255,0) ); // g
    colors.push_back( Scalar(255,0,0) ); // b
    colors.push_back( Scalar(255,255,0) ); // c
    colors.push_back( Scalar(255,0,255) ); // m
    colors.push_back( Scalar(0,255,255) ); // y

    colors.push_back( Scalar(0,0,128) ); // r
    colors.push_back( Scalar(0,128,0) ); // g
    colors.push_back( Scalar(128,0,0) ); // b
    colors.push_back( Scalar(128,128,0) ); // c
    colors.push_back( Scalar(128,0,128) ); // m
    colors.push_back( Scalar(0,128,128) ); // y

    while(first != last)
    {
        const LineSegment & l = *first;
        int g = l.group_id;
        Scalar clr;
        int w;
        if (g == -1)
        {
            clr = Scalar(255,255,255);
            w = 3;
        }
        else
        {
            clr = colors[(g)%12];
            w = 3;
        }
        line(image, cv::Point(l.x1, l.y1), cv::Point(l.x2, l.y2), clr, w);
        circle(image, cv::Point(l.x1, l.y1), 5, clr, -1);
        circle(image, cv::Point(l.x2, l.y2), 5, clr, -1);
        ++first;
    }
}

// Wrapper for find_line_segment_groups accepting OpenCV images of any size
LineSegment * detect_line_groups(
    const Mat & image, int max_size, bool refine, bool use_prosac,
    int num_threads,
    int * n_lines)
{
    Mat image_f = image;
    image.convertTo(image_f, CV_32F, 1/256.0);
    
    float scale;
    image_f = prescale_image(image_f, max_size, scale);

    float * buffer = (float*)image_f.data;
    int w = image_f.cols;
    int h = image_f.rows;
    int stride = w;
    int min_length = float(max(h,w)) / 100.0f;

    LineSegment * lines = find_line_segment_groups(buffer, w, h, stride, min_length, refine, use_prosac, num_threads, n_lines);

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
void homography_from_corners(const ImageTransform & t, float clip, Mat & H, Size & dst_size)
{
    // Find bounding box of the transform
    vector<float> x = {t.top_left.x, t.top_right.x, t.bottom_left.x, t.bottom_right.x};
    vector<float> y = {t.top_left.y, t.top_right.y, t.bottom_left.y, t.bottom_right.y};
    auto xrange = std::minmax_element(x.begin(), x.end());
    auto yrange = std::minmax_element(y.begin(), y.end());

    float xmin = *xrange.first;
    float xmax = *xrange.second;
    float ymin = *yrange.first;
    float ymax = *yrange.second;

    // Center of the transformed image
    float xc = (xmax + xmin) / 2;
    float yc = (ymax + ymin) / 2;

    // New image size - clip is too large
    float width = min(xmax - xmin, t.width * clip);
    float height = min(ymax - ymin, t.height * clip);
    dst_size = Size(int(width), int(height));

    // New zero coordinate
    xmin = xc - 0.5*width;
    ymin = yc - 0.5*height;

    // Calc the transform
    vector<Point2f> src;
    src.push_back(Point2f(0,       0));
    src.push_back(Point2f(t.width, 0));
    src.push_back(Point2f(0,       t.height));
    src.push_back(Point2f(t.width, t.height));

    vector<Point2f> dst;
    dst.push_back(Point2f(t.top_left.x - xmin,     t.top_left.y - ymin));
    dst.push_back(Point2f(t.top_right.x - xmin,    t.top_right.y - ymin));
    dst.push_back(Point2f(t.bottom_left.x - xmin,  t.bottom_left.y - ymin));
    dst.push_back(Point2f(t.bottom_right.x - xmin ,t.bottom_right.y - ymin));

    H = findHomography(src, dst);    
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
    RectificationStrategy h_strategy {RECTIFY};
    RectificationStrategy v_strategy {RECTIFY};
    string filename {""};
    string suffix {"warp"};
    bool refine_lines {false};
    bool prosac_estimator {false};
    int num_threads {-1};
    int max_image_size {1200};
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
        else if (*first == "-r")
        {
            opt.refine_lines = true;
        }
        else if (*first == "-t")
        {
            ++first;
            opt.num_threads = stoi(*first);
        }
        else if (*first == "-p")
        {
            opt.prosac_estimator = true;
        }
        else if (*first == "-m")
        {
            ++first;
            opt.max_image_size = stoi(*first);
            opt.max_image_size = std::min(opt.max_image_size, 2048);
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

    // Detect lines in image
    
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    int n_lines = 0;
    LineSegment * lines = detect_line_groups(image_8uc, opts.max_image_size, opts.refine_lines, opts.prosac_estimator, opts.num_threads, &n_lines);
    // Now we have n_lines segments lines[0] .. lines[n_lines-1]
    // We can modify them, e.g. add user defined lines (remember to assign them to the correct group)
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    clog << "Detection: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " [ms]" << endl;
    


    // Setup parameters for rectification
    RectificationConfig cfg;
    cfg.h_strategy = opts.h_strategy;
    cfg.v_strategy = opts.v_strategy;
    cfg.horizontal_vp_min_distance = 2;

    // Get the locations of corners
    ImageTransform t = compute_rectification_transform(lines, n_lines, image_8uc.cols, image_8uc.rows, cfg);

    ///////////////////////////////////////////////////////////////////////////

    // Calculate transform for image warping
    Mat H;
    Size dst_size;
    homography_from_corners(t, 3.0, H, dst_size);

    Mat image_warped;
    warpPerspective(image_rgb, image_warped, H, dst_size);

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

    release_line_segments(&lines);
}

#
# Roman Juranek <ijuranek@fit.vutbr.cz>
# Faculty of Information Technology, BUT, Brno
#

CXX = g++
CC = gcc
RM = rm
AR = ar


# DEFINES =
# CXXFLAGS = $(DEFINES)  -fPIC -Wall -pedantic -O1

DEFINES = -DEIGEN_FAST_MATH=1 -DEIGEN_NO_DEBUG=1 -DNDEBUG -D_USE_MATH_DEFINES
CXXFLAGS = $(DEFINES) -fPIC -Wall -pedantic -O3 -std=c++11 -fopenmp -ffast-math -fomit-frame-pointer -msse4.2  -mavx -mfma
#CXXFLAGS = $(DEFINES) -fPIC -Wall -pedantic -O3 -std=c++14  -ffast-math -fomit-frame-pointer -msse4.2 -mavx -mfma

EIGEN_INC = -I ../include/eigen
EIGEN_UNSUPPORTED_INC = -I ../include/eigen/unsupported
INCS = $(EIGEN_INC)

OBJS = image.o filter.o geometry.o line_detector.o liblgroup.o ransac.o transform.o threading.o

.PHONY: 

all: shared static autorectify
shared: liblgroup.so
static: liblgroup.a

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCS) $(CV_INCS)  -c $< -o $@

%.s: %.cpp
	$(CXX) $(CXXFLAGS) $(INCS) $(CV_INCS) -S $< -o $@

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCS) $(CV_INCS) -c $< -o $@

%.o: %.c
	$(CC) $(CXXFLAGS) $(INCS) $(CV_INCS) -c $< -o $@

filter.o: filter.cpp
line_detector.o: line_detector.cpp

liblgroup.so: $(OBJS)
	g++ $(OBJS) -shared -Wl,--version-script=exports.map,--soname=$@ -lgomp -o $@

liblgroup.a: $(OBJS)
	ar rcs $@ $^

autorectify: autorectify.o
	g++ -o autorectify $^ -L. -llgroup -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_calib3d

tensortest: tensortest.o filter.o
	g++ -o $@ $^ -L.

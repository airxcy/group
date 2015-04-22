#ifndef UTILS
#define UTILS
#include <vector>
int getLineIdx(std::vector<int>& x_idx,std::vector<int>&  y_idx,int* PointA,int* PointB);
int getLineProp(std::vector<int>& x_idx,std::vector<int>&  y_idx,int* PointA,int* PointB,double linedist);
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}
typedef struct {
	double r;       // percent
	double g;       // percent
	double b;       // percent
} rgb;

typedef struct {
	double h;       // angle in degrees
	double s;       // percent
	double v;       // percent
} hsv;

static hsv      rgb2hsv(rgb in);
static rgb      hsv2rgb(hsv in);
#endif // UTILS


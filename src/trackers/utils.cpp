#include"trackers/utils.h"

int getLineIdx(std::vector<int>& x_idx,std::vector<int>&  y_idx,int* PointA,int* PointB)
{
    /*
    draw a line from PointA to PointB store postions in x_idx y_idx
    return length of the line always >0
    reference Bresenham's line algorithm
    you need to free memory of x_idx y_idx your self
    */
    int idx_N=0,startx=PointA[0],starty=PointA[1],endx=PointB[0],endy=PointB[1];
    int diffx=(endx-startx),diffy=(endy-starty);
    int dx=(diffx > 0) - (diffx < 0), dy=(diffy > 0) - (diffy < 0);
        diffx=diffx*dx,diffy=diffy*dy;
    int x=startx,y=starty;
    int step,incre,err,thresherr;
    int *increter,*steper;
    if(diffx>=diffy)
    {
        err=diffy;
        thresherr=diffx/2;
        increter=&y;
        incre=dy;
        steper=&x;
        step=dx;
        idx_N=diffx;
    }
    else
    {
        err=diffx;
        thresherr=diffy/2;
        increter=&x;
        incre=dx;
        steper=&y;
        step=dy;
        idx_N=diffy;
    }
    int toterr=0,i=0;
    for(i=0;i<idx_N;i++)
    {
        x_idx.push_back(x);
        y_idx.push_back(y);
        (*steper)+=step;
        toterr+=err;
        if((toterr)>=thresherr)
        {
            toterr=toterr-idx_N;
            (*increter)+=incre;
        }
    }
    x_idx.push_back(PointB[0]);
    y_idx.push_back(PointB[1]);
    return idx_N+1;
}
int getLineProp(std::vector<int>& x_idx,std::vector<int>&  y_idx,int* PointA,int* PointB,double linedist)
{
    int xA=PointA[0],yA=PointA[1],xB=PointB[0],yB=PointB[1];
    int linelen = linedist+0.5+1;
    double xstep=(xB-xA)/linedist;
    double ystep=(yB-yA)/linedist;
    for(int i=0;i<linelen;i++)
    {
        int x = xA+i*xstep+0.5;
        int y = yA+i*ystep+0.5;
        x_idx.push_back(x);
        y_idx.push_back(y);
    }
    return linelen;
}


hsv rgb2hsv(rgb in)
{
	hsv         out;
	double      min, max, delta;

	min = in.r < in.g ? in.r : in.g;
	min = min  < in.b ? min : in.b;

	max = in.r > in.g ? in.r : in.g;
	max = max  > in.b ? max : in.b;

	out.v = max;                                // v
	delta = max - min;
	if (max > 0.0) { // NOTE: if Max is == 0, this divide would cause a crash
		out.s = (delta / max);                  // s
	}
	else {
		// if max is 0, then r = g = b = 0              
		// s = 0, v is undefined
		out.s = 0.0;
		out.h = NAN;                            // its now undefined
		return out;
	}
	if (in.r >= max)                           // > is bogus, just keeps compilor happy
		out.h = (in.g - in.b) / delta;        // between yellow & magenta
	else
		if (in.g >= max)
			out.h = 2.0 + (in.b - in.r) / delta;  // between cyan & yellow
		else
			out.h = 4.0 + (in.r - in.g) / delta;  // between magenta & cyan

	out.h *= 60.0;                              // degrees

	if (out.h < 0.0)
		out.h += 360.0;

	return out;
}


rgb hsv2rgb(hsv in)
{
	double      hh, p, q, t, ff;
	long        i;
	rgb         out;

	if (in.s <= 0.0) {       // < is bogus, just shuts up warnings
		out.r = in.v;
		out.g = in.v;
		out.b = in.v;
		return out;
	}
	hh = in.h;
	if (hh >= 360.0) hh = 0.0;
	hh /= 60.0;
	i = (long)hh;
	ff = hh - i;
	p = in.v * (1.0 - in.s);
	q = in.v * (1.0 - (in.s * ff));
	t = in.v * (1.0 - (in.s * (1.0 - ff)));

	switch (i) {
	case 0:
		out.r = in.v;
		out.g = t;
		out.b = p;
		break;
	case 1:
		out.r = q;
		out.g = in.v;
		out.b = p;
		break;
	case 2:
		out.r = p;
		out.g = in.v;
		out.b = t;
		break;

	case 3:
		out.r = p;
		out.g = q;
		out.b = in.v;
		break;
	case 4:
		out.r = t;
		out.g = p;
		out.b = in.v;
		break;
	case 5:
	default:
		out.r = in.v;
		out.g = p;
		out.b = q;
		break;
	}
	return out;
}
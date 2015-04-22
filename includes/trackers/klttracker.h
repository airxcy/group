#ifndef KLTTRACKER_H
#define KLTTRACKER_H
#include "trackers/tracker.h"
//#include "trackers/klt_c/klt.h"

#include <vector>
class KLTtracker : public Tracker
{
public:
    int *isTracking;
	int frameidx;
    int nFeatures,nSearch; /*** get frature number ***/
	unsigned char* preframedata,* bgdata,*curframedata;
    Buff<int> goodNewPts;
    std::vector<FeatBuff> trackBuff;
    FeatPts pttmp;
	ofv ofvtmp;
	float* dirvec;

	/**cuda **/
	Buff<ofv> ofvBuff;
	unsigned char *h_neighbor;
	unsigned int* h_isnewmat,* h_topK;
	float* h_distmat, *h_curvec, *h_group , *h_clrmap,*h_crossDist;
	
	int h_pairN;
	int init(int bsize,int w,int h);
	int selfinit(unsigned char* framedata);
    int updateAframe(unsigned char* framedata,unsigned char* rgbdata,int fidx);
	bool checkTrackMoving(FeatBuff &strk);

	int endTraking();
};
#endif

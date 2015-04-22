#include "trackers/klttracker.h"

#include <cmath>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <cuda.h>
//#include <cublas.h>
#include <cuda_runtime.h>
using namespace cv;
using namespace std;
#define PI 3.14159265

#define persA 0.06
#define persB 40
#define minDist 10
#define TOPK 5
//#define persA 0.01
//#define persB 20
//#define minDist 5

Mat corners,prePts,nextPts,status,eigenvec;
cv::gpu::GoodFeaturesToTrackDetector_GPU detector;
cv::gpu::PyrLKOpticalFlow tracker;
cv::gpu::GpuMat gpuGray, gpuPreGray, gpuCorners, gpuPrePts, gpuNextPts,gpuStatus,gpuEigenvec,gpuDenseX,gpuDenseY,gpuDenseXC,gpuDenseYC;
Mat denseX,denseY,denseRGB;

typedef struct
{
	int i0, i1;
	float correlation;
}ppair, p_ppair;

__device__ int d_pairN[1];
__device__ float maxr[1], maxg[1], maxb[1];
__device__ unsigned char *d_neighbor;
__device__ unsigned int* d_isnewmat, *d_netOrder;
__device__ float* d_distmat, *d_curvec, *d_group, *d_correlation, *d_netUpdate, *d_clrmap, *h_netUpdate, *d_crossDist,*d_topK;
__device__ ofv* d_ofvec;
__device__ ppair* d_pairvec, *h_pairvec;
__device__ unsigned char d_baseclr[6][3]=
{
	{ 0, 255, 0 },
	{ 0, 0, 255 },
	{ 255, 255, 0 },
	{ 255, 0, 255 },
	{ 0, 255, 255 },
	{ 255, 0, 0 },
};
__global__ void crossDist(unsigned int* dst,float* vertical,float* horizon,int h,int w)
{
	int x = threadIdx.x,y=blockIdx.x;
	if (x < w&&y<h)
	{
		float xv = vertical[y * 2], yv = vertical[y*2+1],xh=horizon[x*2],yh=horizon[x*2+1];
		float dx = xv - xh, dy = yv - yh;
		float dist = abs(dx) + abs(dy);
		if (dist < minDist)
			atomicAdd(dst + y, 1);
	}
}
__global__ void searchNeighbor(unsigned char* d_neighbor, ofv* d_ofvec , ppair* d_pairvec,unsigned int* d_netOrder, int nFeatures)
{
	int r = blockIdx.x, c = threadIdx.x;
	if (r < c)
	{
		float dx = abs(d_ofvec[r].x1 - d_ofvec[c].x1), dy = abs(d_ofvec[r].y1 - d_ofvec[c].y1);
		int yidx = d_ofvec[r].idx, xidx = d_ofvec[c].idx;
		float dist = sqrt(dx*dx + dy*dy);
		float xmid = (d_ofvec[r].x1 + d_ofvec[c].x1) / 2, ymid = (d_ofvec[r].y1 + d_ofvec[c].y1) / 2;
		if (dx < ymid*(persA)+persB && dy < ymid*(persA*1.5) + persB*1.5)
		{
			d_neighbor[yidx*nFeatures + xidx] = 1;
			d_neighbor[xidx*nFeatures + yidx] = 1;
			float vx0 = d_ofvec[r].x1 - d_ofvec[r].x0, vx1 = d_ofvec[c].x1 - d_ofvec[c].x0,
				vy0 = d_ofvec[r].y1 - d_ofvec[r].y0, vy1 = d_ofvec[c].y1 - d_ofvec[c].y0;
			float norm0 = sqrt(vx0*vx0 + vy0*vy0), norm1 = sqrt(vx1*vx1 + vy1*vy1);
			float cosine = (vx0*vx1 + vy0*vy1) / norm0 / norm1;
			float cor = cosine / (dist / 10 + 0.1);
			//if (cosine > 0.5
				ppair tmppair;
				tmppair.i0 = yidx, tmppair.i1 = xidx, tmppair.correlation = cor;
				int arrpos = atomicAdd(d_pairN, 1);
				memcpy(d_pairvec + arrpos, &tmppair,sizeof(ppair));
				atomicAdd(d_netOrder + yidx,1);
				atomicAdd(d_netOrder + xidx, 1);
		}
		/*
		else
			d_neighbor[yidx*nFeatures + xidx] = 0;
		*/
	}
}
__global__ void calUpdate(float* d_group, ppair* d_pairvec, float* d_netUpdate)
{
	int nPair = gridDim.x,nFeatures = blockDim.x;
	int ipair = blockIdx.x, idim = threadIdx.x;
	int i0 = d_pairvec[ipair].i0, i1 = d_pairvec[ipair].i1;
	float cor = d_pairvec[ipair].correlation;
	//printf("%f\n", cor);
	float update0 = d_group[i1*nFeatures + idim] * cor;
	float update1 = d_group[i0*nFeatures + idim] * cor;
	
	atomicAdd(d_netUpdate + i0*nFeatures + idim, update0);
	atomicAdd(d_netUpdate + i1*nFeatures + idim, update1);
}
__global__ void updateNet(float* d_group, float* d_netUpdate,unsigned int* d_netOrder)
{
	int idx = blockIdx.x, nFeatures = blockDim.x;
	int dim = threadIdx.x;
	int order = d_netOrder[idx];
	if (order > 0)
	{
		float newval = d_netUpdate[idx*nFeatures+dim]/order;
		float oldval = d_group[idx*nFeatures + dim];
		d_group[idx*nFeatures + dim] = (oldval + newval) / 2;
	}
}
__global__ void inCross(float* d_group,float* d_crossDist)
{
	int nFeatures = blockDim.x;
	int i0 = blockIdx.x, i1 = threadIdx.x;
	float val = 0;
	__shared__  float group0[1000];
	__shared__  float group1[1000];
	memcpy(group0, d_group + i0*nFeatures,nFeatures*sizeof(float));
	memcpy(group1, d_group + i1*nFeatures, nFeatures*sizeof(float));
	for (int i = 0; i < nFeatures; i++)
	{
		//val += d_group[i0*nFeatures + i] * d_group[i1*nFeatures + i];
		val += group0[i] * group1[i];
	}
	d_crossDist[i0*nFeatures + i1] = val;
}
__global__ void genClr(float* d_clrmap, float* d_group)
{
	int nFeatures = blockDim.x;
	int idx = blockIdx.x, idim = threadIdx.x;
	unsigned char r = d_baseclr[idim % 6][0], g = d_baseclr[idim % 6][1], b = d_baseclr[idim % 6][2];
	float val = d_group[idx*nFeatures + idim]/nFeatures;
	atomicAdd(d_clrmap + idx * 3, r*val);
	atomicAdd(d_clrmap + idx * 3+1, g*val);
	atomicAdd(d_clrmap + idx * 3+2, b*val);
}

int KLTtracker::init(int bsize,int w,int h)
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
		std::cout << "maxgridDim" << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << std::endl;
		std::cout<<"maxThreadsPerBlock:"<<prop.maxThreadsPerBlock<<std::endl;
		std::cout << prop.major << "," << prop.minor << std::endl;
	}

    nFeatures = 1000;
    nSearch=3000;
	trackBuff = std::vector<FeatBuff>(nFeatures);
    isTracking=new int[nFeatures];
	for (int i=0;i<nFeatures;i++)
	{
		trackBuff[i].init(1,100);
        isTracking[i]=0;
	}
	frame_width = w;
	frame_height = h;
	frameidx=0;
	dirvec = new float[nFeatures];
	memset(dirvec, 0, nFeatures*sizeof(float));

    goodNewPts.init(1,nSearch);
    detector= gpu::GoodFeaturesToTrackDetector_GPU(nSearch,0.0001,3,3);
    tracker = gpu::PyrLKOpticalFlow();
    tracker.winSize=Size(3,3);
    tracker.maxLevel=3;
    tracker.iters=10;
    gpuGray=gpu::GpuMat(frame_height, frame_width, CV_8UC1 );
    gpuPreGray=gpu::GpuMat(frame_height, frame_width, CV_8UC1 );

	cudaMalloc(&d_isnewmat, nSearch*sizeof(unsigned int));
	h_isnewmat = (unsigned int*)malloc(nSearch*sizeof(unsigned int));


	h_curvec = (float*)malloc(nFeatures*2*sizeof(float));
	cudaMalloc(&d_curvec, nFeatures * 2 * sizeof(float));
	cudaMalloc(&d_ofvec, nFeatures* sizeof(ofv));
	ofvBuff.init(1, nFeatures);

	cudaMalloc(&d_neighbor, nFeatures*nFeatures);
	h_neighbor = (unsigned char*)malloc(nFeatures*nFeatures);
	memset(h_neighbor, 0, nFeatures*nFeatures);

	cudaMalloc(&d_group, nFeatures*nFeatures*sizeof(float));
	h_group = (float *)malloc(nFeatures*nFeatures*sizeof(float));
	cudaMalloc(&d_netOrder, nFeatures*sizeof(unsigned int));
	//cudaMalloc(&d_correlation, nFeatures*nFeatures*sizeof(float));
	cudaMalloc(&d_pairvec, nFeatures*sizeof(ppair));
	h_pairvec = (ppair *)malloc(nFeatures*sizeof(ppair));
	cudaMalloc(&d_netUpdate, nFeatures*nFeatures*sizeof(float));
	h_netUpdate = (float*)malloc(nFeatures*nFeatures*sizeof(float));
	h_pairN = 0;
	cudaMemcpyToSymbol(d_pairN, &h_pairN, sizeof(int));
	cudaMalloc(&d_crossDist, nFeatures*nFeatures*sizeof(float));
	h_crossDist = (float *)malloc(nFeatures*nFeatures*sizeof(float));

	cudaMalloc(&d_clrmap, nFeatures * 3 * sizeof(float));
	h_clrmap = (float*)malloc(nFeatures * 3 * sizeof(float));
	//cudaMalloc(&d_topK, nFeatures * TOPK * sizeof(float));
	h_topK = (unsigned int*)malloc(nFeatures * TOPK * sizeof(unsigned int*));
	memset(h_topK, 0, nFeatures * TOPK * sizeof(unsigned int*));
	std::cout << "inited" << std::endl;
	gt_inited = false;

	return 1;
}
int KLTtracker::selfinit(unsigned char* framedata)
{
	curframedata=framedata;
    Mat curframe(frame_height,frame_width,CV_8UC1,framedata);
    gpuGray.upload(curframe);
    gpuPreGray.upload(curframe);
    detector(gpuGray, gpuCorners);
    gpuCorners.download(corners);
    gpuCorners.copyTo(gpuPrePts);
    for (int k = 0; k < nFeatures; k++)
    {
        Vec2f p = corners.at<Vec2f>(k);
		pttmp.x = p[0];//(PntT)(p[0] + 0.5);
		pttmp.y = p[1];//(PntT)(p[1]+ 0.5);
        pttmp.t = frameidx;
        trackBuff[k].updateAFrame(&pttmp);
        isTracking[k]=1;
		memset(h_group + k*nFeatures, 0, nFeatures*sizeof(float));
		h_group[k*nFeatures + k] = 1;
		h_curvec[k * 2] = trackBuff[k].cur_frame_ptr->x;
		h_curvec[k * 2 + 1] = trackBuff[k].cur_frame_ptr->y;
    }
	return true;
}
bool KLTtracker::checkTrackMoving(FeatBuff &strk)
{
	bool isTrkValid = true;
    int Movelen=7,minlen=5,startidx=max(strk.len-Movelen,0);
    if(strk.len>Movelen)
    {
        double maxdist = .0, dtmp = .0,totlen=.0;
		FeatPts* aptr = strk.getPtr(startidx), *bptr = aptr;
        PntT xa=aptr->x,ya=aptr->y,xb=strk.cur_frame_ptr->x,yb=strk.cur_frame_ptr->y;
        double displc=sqrt( pow(xb-xa, 2.0) + pow(yb-ya, 2.0));
        if((strk.len -startidx)*0.2>displc)
        {
            isTrkValid = false;
        }

    }
	return isTrkValid;
}

int KLTtracker::updateAframe(unsigned char* framedata, unsigned char* rgbdata, int fidx)
{
    frameidx=fidx;
	curframedata=framedata;
    gpuGray.copyTo(gpuPreGray);
	//gpuPreGray.data = gpuGray.data;
    Mat curframe(frame_height,frame_width,CV_8UC1,framedata);
    gpuGray.upload(curframe);

    tracker.sparse(gpuPreGray, gpuGray, gpuPrePts, gpuNextPts, gpuStatus, &gpuEigenvec);
    gpuStatus.download(status);
    gpuNextPts.download(nextPts);
    gpuPrePts.download(prePts);
    detector(gpuGray, gpuCorners);
    gpuCorners.download(corners);

	cudaMemcpy(d_curvec, h_curvec, nFeatures*2*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(d_isnewmat, 0, nSearch*sizeof(unsigned int));
	crossDist<<<nSearch, nFeatures>>>(d_isnewmat, (float *)gpuCorners.data, d_curvec, nSearch, nFeatures);
	cudaMemcpy(h_isnewmat, d_isnewmat, nSearch*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	goodNewPts.clear();
    for(int i=0;i<nSearch;i++)
    {
		if (h_isnewmat[i] == 0)
        {
            goodNewPts.updateAFrame(&i);
        }
    }
	ofvBuff.clear();
    int addidx=0,counter=0;
	for (int k = 0; k < nFeatures; k++)
	{
        int statusflag = status.at<int>(k);
        Vec2f trkp = nextPts.at<Vec2f>(k);
        bool lost=false;
        if ( statusflag)
		{
            Vec2f pre=prePts.at<Vec2f>(k),cur=nextPts.at<Vec2f>(k);
            int prex=trackBuff[k].cur_frame_ptr->x, prey=trackBuff[k].cur_frame_ptr->y;
			pttmp.x = trkp[0];
			pttmp.y = trkp[1];
            pttmp.t = frameidx;
            trackBuff[k].updateAFrame(&pttmp);
            double trkdist=abs(prex-pttmp.x)+abs(prey-pttmp.y),ofdist=abs(pre[0]-cur[0])+abs(pre[1]-cur[1]);
			dirvec[k] = 0.5*dirvec[k] + 0.5*sgn(pttmp.y-prey);
            isTracking[k]=1;
			bool isMoving=checkTrackMoving(trackBuff[k]);
			if (!isMoving||(trackBuff[k].len>1 && trkdist>10))
			{
                lost=true;
                isTracking[k]=0;
			}
		}
        else
        {
            counter++;
            lost=true;
            isTracking[k]=0;
		}
        if(lost)
        {
            trackBuff[k].clear();
			dirvec[k] = 0;
            if(addidx<goodNewPts.len)
            {
                int newidx=*(goodNewPts.getPtr(addidx++));
                Vec2f cnrp = corners.at<Vec2f>(newidx);
				pttmp.x = cnrp[0];
				pttmp.y = cnrp[1];
                pttmp.t = frameidx;
                trackBuff[k].updateAFrame(&pttmp);
                nextPts.at<Vec2f>(k)=cnrp;
                isTracking[k]=1;
				memset(h_group + k*nFeatures, 0, nFeatures*sizeof(float));
				h_group[k*nFeatures + k] = 1;
            }
        }
		else
		{
			if (trackBuff[k].len > 8)
			{
				ofvtmp.x0 = trackBuff[k].getPtr(trackBuff[k].len - 5)->x;
				ofvtmp.y0 = trackBuff[k].getPtr(trackBuff[k].len - 5)->y;
				ofvtmp.x1 = trackBuff[k].cur_frame_ptr->x;
				ofvtmp.y1 = trackBuff[k].cur_frame_ptr->y;
				ofvtmp.len = trackBuff[k].len;
				ofvtmp.idx = k;
				ofvBuff.updateAFrame(&ofvtmp);
			}
		}
		h_curvec[k * 2] = trackBuff[k].cur_frame_ptr->x;
		h_curvec[k * 2 + 1] = trackBuff[k].cur_frame_ptr->y;
	}
	
	if (ofvBuff.len > 0)
	{
		h_pairN = 0;
		cudaMemset(d_ofvec, 0, nFeatures* sizeof(ofv));
		cudaMemcpy(d_ofvec, ofvBuff.data, ofvBuff.len*sizeof(ofv), cudaMemcpyHostToDevice);
		cudaMemset(d_neighbor, 0, nFeatures*nFeatures);
		cudaMemset(d_netOrder, 0, nFeatures*sizeof(unsigned int));
		cudaMemcpyToSymbol(d_pairN, &h_pairN, sizeof(int));
		searchNeighbor << <ofvBuff.len, ofvBuff.len >> >(d_neighbor, d_ofvec, d_pairvec,d_netOrder, nFeatures);
		cudaMemcpy(h_pairvec, d_pairvec, nFeatures*sizeof(ppair), cudaMemcpyDeviceToHost);

		cudaMemcpy(h_neighbor, d_neighbor, nFeatures*nFeatures, cudaMemcpyDeviceToHost);
		cudaMemcpyFromSymbol(&h_pairN, d_pairN, sizeof(int));
		cudaMemcpy(d_group, h_group, nFeatures*nFeatures*sizeof(float), cudaMemcpyHostToDevice);
		std::cout << "h_pairN:" << h_pairN << std::endl;
		//for (int i = 0; i < nFeatures; i++)
		//{

		//}
		cudaMemset(d_netUpdate, 0, nFeatures*nFeatures*sizeof(float));
		calUpdate <<<h_pairN, nFeatures>>>(d_group, d_pairvec, d_netUpdate);
		cudaMemcpy(h_netUpdate, d_netUpdate, nFeatures*nFeatures*sizeof(float), cudaMemcpyDeviceToHost);
		updateNet <<<nFeatures,nFeatures>>>(d_group, d_netUpdate, d_netOrder);
		cudaMemcpy(h_group, d_group, nFeatures*nFeatures*sizeof(float), cudaMemcpyDeviceToHost);
		/*
		float maxval;
		for (int i = 0; i < nFeatures; i++)
		{
			for (int j = 0; j < nFeatures; j++)
			{
				int ind = (maxval>h_group[i*nFeatures + j]);
				maxval = ind*maxval + (1 - ind)*h_group[i*nFeatures + j];
			}
		}
		std::cout <<maxval<< std::endl;
		*/
		inCross << <nFeatures, nFeatures >> >(d_group, d_crossDist);
		cudaMemcpy(h_crossDist, d_crossDist, nFeatures*nFeatures*sizeof(float), cudaMemcpyDeviceToHost);
		
		for (int i = 0; i < nFeatures; i++)
		{
			std::cout << h_crossDist[i*nFeatures + i] << std::endl;
		}
		
		
		/*
		std::cout << std::endl;
		cudaMemset(d_clrmap, 0, 3 * nFeatures*sizeof(float));
		genClr << <nFeatures, nFeatures >> >(d_clrmap, d_group);
		cudaMemcpy(h_clrmap, d_clrmap, 3*nFeatures*sizeof(float), cudaMemcpyDeviceToHost);
		float maxr = 0, maxg = 0, maxb = 0;
		for (int j = 0; j < nFeatures; j++)
		{
			
			int ind = (maxr>h_clrmap[j*3]);
			maxr = ind*maxr + (1 - ind)*h_clrmap[j * 3];
			ind = (maxg>h_clrmap[j * 3+1]);
			maxg = ind*maxg + (1 - ind)*h_clrmap[j * 3+1];
			ind = (maxb>h_clrmap[j * 3+2]);
			maxb = ind*maxb + (1 - ind)*h_clrmap[j * 3+2];
		}
		for (int j = 0; j < nFeatures; j++)
		{

			h_clrmap[j * 3] /= maxr/255;
			h_clrmap[j * 3+1] /= maxg/255;
			h_clrmap[j * 3+2] /= maxb/255;
			//std::cout << h_clrmap[j * 3] << "," << h_clrmap[j * 3 + 1] << "," << h_clrmap[j * 3 + 2] << "," << std::endl;
		}
		*/
	}
	
    gpuPrePts.upload(nextPts);
	return 1;
}

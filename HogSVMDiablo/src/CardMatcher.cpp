#define _CRT_SECURE_NO_DEPRECATE

#include "CardMatcher.h"

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>

#include <iomanip>
#include <io.h>

#include "logc.h"

using namespace std;
using namespace cv;

int CardMatcher::setModelSIFT(char * modelPath)
{
	char * imgName = 0;
	char * flagsPath = (char *)malloc(strlen(modelPath) + strlen("\\flags\\*.*") + 1);

	strcpy(flagsPath, modelPath);
	strcat(flagsPath, "\\flags\\*.*");

	long handle;
	struct _finddata_t fileinfo;
	handle = _findfirst(flagsPath, &fileinfo);
	if (-1 == handle)
	{
		LogWrite(ERROR, "%s", "ERROR: No Files in flagsPath.");
		return -1;
	}

	while (!_findnext(handle, &fileinfo))
	{
		imgName = fileinfo.name;
		while (strcmp(imgName, ".") == 0 || strcmp(imgName, "..") == 0)
		{
			_findnext(handle, &fileinfo);
			imgName = fileinfo.name;
		}

		char * srcName = (char *)malloc(strlen(modelPath) + strlen("\\flags\\") + strlen(imgName));

		//处理NUMBERS
		strcpy(srcName, modelPath);
		strcat(srcName, "\\flags\\");
		strcat(srcName, imgName);

		Mat img = imread(srcName, 0);
		//resize(img, img, Size(50, 50));
		SiftFeatureDetector siftDtc;
		vector<KeyPoint> keyPoint;

		siftDtc.detect(img, keyPoint);
		SiftDescriptorExtractor extractor;
		Mat descriptor;
		extractor.compute(img, keyPoint, descriptor);

		String key = imgName;
		this->flagSIFTDescriptors.insert(make_pair(key.substr(0, strlen(imgName)-4), descriptor));

	}
	_findclose(handle);


	char * numbersPath = (char *)malloc(strlen(modelPath) + strlen("\\numbers\\*.*") + 1);
	strcpy(numbersPath, modelPath);
	strcat(numbersPath, "\\numbers\\*.*");

	handle = _findfirst(numbersPath, &fileinfo);
	if (-1 == handle)
	{
		LogWrite(ERROR, "%s", "ERROR: No Files in flagsPath.");
		return -1;
	}

	while (!_findnext(handle, &fileinfo))
	{
		imgName = fileinfo.name;
		while (strcmp(imgName, ".") == 0 || strcmp(imgName, "..") == 0)
		{
			_findnext(handle, &fileinfo);
			imgName = fileinfo.name;
		}

		char * srcName = (char *)malloc(strlen(modelPath) + strlen("\\numbers\\") + strlen(imgName));

		//处理NUMBERS
		strcpy(srcName, modelPath);
		strcat(srcName, "\\numbers\\");
		strcat(srcName, imgName);

		Mat img = imread(srcName, 0);
		//resize(img, img, Size(50, 50));

		SiftFeatureDetector siftDtc;
		vector<KeyPoint> keyPoint;

		siftDtc.detect(img, keyPoint);
		SiftDescriptorExtractor extractor;
		Mat descriptor;
		extractor.compute(img, keyPoint, descriptor);

		String key = imgName;
		this->numberSIFTDescriptors.insert(make_pair(key.substr(0, strlen(imgName) - 4), descriptor));

	}
	_findclose(handle);

	return 1;

}

Card* CardMatcher::Mat2Card(Mat img)
{
	Card *card = (Card*)malloc(sizeof(Card));
	FlannBasedMatcher matcher;
	
	//resize(img, img, Size(600,600));
	
	SiftFeatureDetector siftDtc;
	vector<KeyPoint> keyPoint;

	siftDtc.detect(img, keyPoint);
	SiftDescriptorExtractor extractor;
	Mat descriptor1;
	extractor.compute(img, keyPoint, descriptor1);

	float MinDistance = 10000000;
	char* flag = NULL;
	for (int i = 0; i < 4; i++)
	{
		vector<DMatch> matches;
		Mat descriptor2 = this->flagSIFTDescriptors[FLAGS[i]];

		matcher.match(descriptor2, descriptor1, matches);
		
		double maxDist = 0;
		double minDist = 100;

		for (int i = 0; i < descriptor2.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < minDist)
				minDist = dist;
			if (dist > maxDist)
				maxDist = dist;
		}
		
		vector<DMatch> goodMatches;
		
		float meanDist = 0;
		int count = 0;
		for (int i = 0; i < descriptor2.rows; i++)
		{
			if (matches[i].distance <= 5 * minDist)
			{
				goodMatches.push_back(matches[i]);
				meanDist = (float)(meanDist*count + matches[i].distance) / (++count);
			}
				
		}

		//findHomography(keyPoint, keyPoint2, CV_RANSAC, 5.0, matches);
		
		if (MinDistance > meanDist)
		{
			MinDistance = meanDist;
			flag = FLAGS[i];
		}
	}

	if (NULL == flag)
	{
		return NULL;
	}
	card->flag = flag;
	
	char* number = NULL;

	for (int i = 0; i < 13; i++)
	{
		vector<DMatch> matches;
		Mat descriptor2 = this->numberSIFTDescriptors[NUMBERS[i]];
		matcher.match(descriptor2, descriptor1, matches);
		
		double maxDist = 0;
		double minDist = 100;

		for (int i = 0; i < descriptor2.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < minDist)
				minDist = dist;
			if (dist > maxDist)
				maxDist = dist;
		}

		vector<DMatch> goodMatches;

		float meanDist = 0;
		int count = 0;
		for (int i = 0; i < descriptor2.rows; i++)
		{
			if (matches[i].distance <= 5 * minDist)
			{
				goodMatches.push_back(matches[i]);
				meanDist = (float)(meanDist*count + matches[i].distance) / (++count);
			}

		}
		//Mat H = findHomography(descriptor2, descriptor1, CV_RANSAC);
		//perspectiveTransform(obj_corners, scene_corners, H);
		//meanDist = Hausdorff.compute(scene_edge, dst_edge);

		if (MinDistance > meanDist)
		{
			MinDistance = meanDist;
			number = NUMBERS[i];
		}
	}
	
	if (NULL == number)
	{
		return NULL;
	}
	card->number = number;

	return card;
}
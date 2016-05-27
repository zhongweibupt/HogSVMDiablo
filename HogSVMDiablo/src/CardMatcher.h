#ifndef CARDMATCHER_H_
#define CARDMATCHER_H_

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/objdetect/objdetect.hpp>  

using namespace std;
using namespace cv;

static char* FLAGS[] = { "spades", "hearts", "box", "plum" };
static char* NUMBERS[] = { "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K" };

typedef struct Card
{
	char * flag;
	char * number;
}Card;

class CardMatcher
{
private:
	map<string, Mat> flagSIFTDescriptors;
	map<string, Mat> numberSIFTDescriptors;

public:
	int setModelSIFT(char * modelPath);
	Card* Mat2Card(Mat img);
};

#endif /* CARDMATCHER_H_ */
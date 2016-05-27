#define _CRT_SECURE_NO_DEPRECATE

#include <iostream>  
#include <fstream>  
#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/objdetect/objdetect.hpp>  
#include <opencv2/ml/ml.hpp>

#include <iomanip>
#include <io.h>
#include <malloc.h>

#include "CardMatcher.h"
#include "logc.h"


using namespace std;
using namespace cv;

#define PosSamNO 400    //正样本个数，越大越好，至少在1000以上  
#define NegSamNO 100    //负样本个数  越大越好，至少在1000以上 

#define PosSam_PATH ".\\images\\train\\1-1\\"
#define NegSam_PATH ".\\images\\train\\0-1\\"

#define TRAIN false    //是否进行训练,true表示重新训练，false表示读取xml文件中的SVM模型  
#define IMAGE_SCALING true   //默认true，训练时，是否将样本图片缩放成80*100大小  

//HardExample：负样本个数。如果HardExampleNO大于0，表示处理完初始负样本集后，继续处理HardExample负样本集。  
//不使用HardExample时必须设置为0，因为特征向量矩阵和特征类别矩阵的维数初始化时用到这个值  
#define HardExampleNO 0 

class MySVM : public CvSVM
{
public:
	//获得SVM的决策函数中的alpha数组  
	double * get_alpha_vector()
	{
		return this->decision_func->alpha;
	}

	//获得SVM的决策函数中的rho参数,即偏移量  
	float get_rho()
	{
		return this->decision_func->rho;
	}

	int get_alpha_count()
	{
		return this->sv_total;
	}

	int get_sv_dim()
	{
		return this->var_all;
	}

	int get_sv_count()
	{
		return this->decision_func->sv_count;
	}

	double* get_alpha()
	{
		return this->decision_func->alpha;
	}

	float** get_sv()
	{
		return this->sv;
	}

};


int loadSVM(MySVM *svm, char * xmlName)
{
	svm->load(xmlName);
	return 1;
}

int trainSVM(MySVM *svm)
{
	HOGDescriptor hog(Size(100, 100), Size(20, 20), Size(10, 10), Size(10, 10), 9);//HOG检测器，用来计算HOG描述子的  
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定  

	Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数      
	Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示正样本，-1表示负样本 

	char * imgName = 0;
	char * temp = (char *)malloc(strlen(PosSam_PATH) + 4);

	//处理正样本
	strcpy(temp, PosSam_PATH);
	strcat(temp, "*.*");
	
	long handle;
	struct _finddata_t fileinfo;
	handle = _findfirst(temp, &fileinfo);
	if (-1 == handle)
	{
		LogWrite(ERROR, "%s", "ERROR: No Files in positive samples' folder.");
		return -1;
	}

	LogWrite(INF, "%s", "INFO: Processing positive samples...");

	int numPos = 0;
	while (!_findnext(handle, &fileinfo))
	{
		imgName = fileinfo.name;
		while (strcmp(imgName, ".") == 0 || strcmp(imgName, "..") == 0)
		{
			_findnext(handle, &fileinfo);
			imgName = fileinfo.name;
		}
		
		char * srcName = (char *)malloc(strlen(PosSam_PATH) + strlen(imgName));

		//处理正样本
		strcpy(srcName, PosSam_PATH);
		strcat(srcName, imgName);

		Mat src = imread(srcName, 0);//读取图片  
		if (IMAGE_SCALING)
			resize(src, src, Size(100, 100));
		cout << srcName << endl;

		vector<float> descriptors;//HOG描述子向量  
		hog.compute(src, descriptors, Size(10, 10));//计算HOG描述子，检测窗口移动步长(8,8)  
		//cout<<"描述子维数："<<descriptors.size()<<endl;  

		//处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵  
		if (0 == numPos)
		{
			DescriptorDim = descriptors.size();//HOG描述子的维数  
			//初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat  
			sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
			//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示正样本，0表示负样本
			sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
		}

		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat  
		for (int i = 0; i<DescriptorDim; i++)
			sampleFeatureMat.at<float>(numPos, i) = descriptors[i];//第num个样本的特征向量中的第i个元素  
		sampleLabelMat.at<float>(numPos, 0) = 1;//正样本类别为1

		numPos++;
	}
	_findclose(handle);


	//处理负样本
	temp = (char *)malloc(strlen(NegSam_PATH) + 4);
	strcpy(temp, NegSam_PATH);
	strcat(temp, "*.*");

	handle = _findfirst(temp, &fileinfo);
	if (-1 == handle)
	{
		LogWrite(ERROR, "%s", "ERROR: No Files in negative samples' folder.");
		return -1;
	}

	LogWrite(INF, "%s", "INFO: Processing negative samples...");

	int numNeg = 0;
	while (!_findnext(handle, &fileinfo))
	{
		imgName = fileinfo.name;
		while (strcmp(imgName, ".") == 0 || strcmp(imgName, "..") == 0)
		{
			_findnext(handle, &fileinfo);
			imgName = fileinfo.name;
		}

		char * srcName = (char *)malloc(strlen(NegSam_PATH) + strlen(imgName));

		//处理负样本
		strcpy(srcName, NegSam_PATH);
		strcat(srcName, imgName);

		Mat src = imread(srcName, 0);//读取图片  
		if (IMAGE_SCALING)
			resize(src, src, Size(100, 100));
		cout << srcName << endl;

		vector<float> descriptors;//HOG描述子向量  
		hog.compute(src, descriptors, Size(10, 10));//计算HOG描述子，检测窗口移动步长(8,8)  
		//cout<<"描述子维数："<<descriptors.size()<<endl;  

		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat  
		for (int i = 0; i<DescriptorDim; i++)
			sampleFeatureMat.at<float>(numNeg + PosSamNO, i) = descriptors[i];//第num个样本的特征向量中的第i个元素  
		sampleLabelMat.at<float>(numNeg + PosSamNO, 0) = -1;//负样本类别为0

		numNeg++;
	}
	_findclose(handle);

	//训练SVM分类器  
	//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代  
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01  
	CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
	cout << "Training SVM classifier..." << endl;
	svm->train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//训练分类器  
	cout << "Completed!" << endl;
	svm->save("SVM_HOG.xml");//将训练好的SVM模型保存为xml文件  


	Mat testImg = imread("C:\\Users\\zhwei\\Desktop\\Diablo\\images\\test\\test_15.jpg");
	if (IMAGE_SCALING)
		resize(testImg, testImg, Size(100, 100));
	vector<float> descriptor;
	hog.compute(testImg, descriptor, Size(10, 10));//计算HOG描述子，检测窗口移动步长(8,8)  
	Mat testFeatureMat = Mat::zeros(1, descriptor.size(), CV_32FC1);//测试样本的特征向量矩阵  
	//将计算好的HOG描述子复制到testFeatureMat矩阵中  
	for (int i = 0; i<descriptor.size(); i++)
		testFeatureMat.at<float>(0, i) = descriptor[i];

	//用训练好的SVM分类器对测试图片的特征向量进行分类  
	int result = svm->predict(testFeatureMat);//返回类标  
	cout << "分类结果：" << result << endl;

}

int setHog(HOGDescriptor * myHOG, MySVM * svm)
{
	/*************************************************************************************************
	线性SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;
	将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个列向量。之后，再该列向量的最后添加一个元素rho。
	如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），
	就可以利用你的训练样本训练出来的分类器进行行人检测了。
	***************************************************************************************************/
	int DescriptorDim = svm->get_var_count();//特征向量的维数，即HOG描述子的维数  
	int supportVectorNum = svm->get_support_vector_count();//支持向量的个数 
	cout << DescriptorDim << endl;
	cout << "支持向量个数：" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数  
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵  
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果  

	//将支持向量的数据复制到supportVectorMat矩阵中  
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm->get_support_vector(i);//返回第i个支持向量的数据指针  
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";  
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//将alpha向量的数据复制到alphaMat中  
	double * pAlphaData = svm->get_alpha_vector();//返回SVM的决策函数中的alpha向量  
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//计算-(alphaMat * supportVectorMat),结果放到resultMat中  
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？  
	resultMat = -1 * alphaMat * supportVectorMat;

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子  
	vector<float> myDetector;
	//将resultMat中的数据复制到数组myDetector中  
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//最后添加偏移量rho，得到检测子  
	myDetector.push_back(svm->get_rho());
	cout << "检测子维数：" << myDetector.size() << endl;
	//设置HOGDescriptor的检测子  
	
	myHOG->setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());  

	//保存检测子参数到文件  
	ofstream fout("HOGDetectorForOpenCV.txt");
	for (int i = 0; i<myDetector.size(); i++)
	{
		fout << myDetector[i] << endl;
	}
	return 1;
}

int detectCard(char * imgName, HOGDescriptor *myHOG)
{
	Mat src = imread(imgName, 0);
	vector<Rect> found, found_filtered;//矩形框数组  
	cout << "进行多尺度检测" << endl;
	myHOG->detectMultiScale(src, found, 0, Size(20, 20), Size(0, 0), 1.05, 2);//对图片进行多尺度检测  
	cout << "检测出目标个数：" << found.size() << endl;

	//找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中  
	for (int i = 0; i < found.size(); i++)
	{
		Rect r = found[i];
		int j = 0;
		for (; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
	}

	//画矩形框
	char * modelPath = ".\\images\\model";
	CardMatcher matcher;
	matcher.setModelSIFT(modelPath);

	for (int i = 0; i<found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];
		//r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width);
		//r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height);
		rectangle(src, r.tl(), r.br(), Scalar(0, 255, 0), 3);

		Mat imgROI = src(r);
		
		Card* card = (Card*)malloc(sizeof(Card));
		card = matcher.Mat2Card(imgROI);

		if (card != NULL)
		{
			cout << "Card:" << endl;
			cout << card->flag << " " << card->number << endl;
		}
		imshow("ROI", imgROI);
		waitKey(0);
	}

	imwrite("ImgProcessed.jpg", src);
	namedWindow("src", 0);
	imshow("src", src);
	waitKey();//注意：imshow之后必须加waitKey，否则无法显示图像  


	/******************读入单个64*128的测试图并对其HOG描述子进行分类*********************/
	////读取测试图片(64*128大小)，并计算其HOG描述子  
	////Mat testImg = imread("person014142.jpg");  
	//Mat testImg = imread("noperson000026.jpg");  
	//vector<float> descriptor;  
	//hog.compute(testImg,descriptor,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)  
	//Mat testFeatureMat = Mat::zeros(1,3780,CV_32FC1);//测试样本的特征向量矩阵  
	////将计算好的HOG描述子复制到testFeatureMat矩阵中  
	//for(int i=0; i<descriptor.size(); i++)  
	//  testFeatureMat.at<float>(0,i) = descriptor[i];  

	////用训练好的SVM分类器对测试图片的特征向量进行分类  
	//int result = svm.predict(testFeatureMat);//返回类标  
	//cout<<"分类结果："<<result<<endl;  

	return 1;
}

int main()
{
	char * imgName = "test.jpg";

	MySVM svm ;
	//trainSVM(&svm);

	loadSVM(&svm, "SVM_HOG.xml");

	HOGDescriptor myHog(Size(100, 100), Size(20, 20), Size(10, 10), Size(10, 10), 9);
	setHog(&myHog, &svm);
	
	detectCard(imgName, &myHog);
	return 1;
}
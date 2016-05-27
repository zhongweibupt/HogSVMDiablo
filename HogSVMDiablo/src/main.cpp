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

#define PosSamNO 400    //������������Խ��Խ�ã�������1000����  
#define NegSamNO 100    //����������  Խ��Խ�ã�������1000���� 

#define PosSam_PATH ".\\images\\train\\1-1\\"
#define NegSam_PATH ".\\images\\train\\0-1\\"

#define TRAIN false    //�Ƿ����ѵ��,true��ʾ����ѵ����false��ʾ��ȡxml�ļ��е�SVMģ��  
#define IMAGE_SCALING true   //Ĭ��true��ѵ��ʱ���Ƿ�����ͼƬ���ų�80*100��С  

//HardExample�����������������HardExampleNO����0����ʾ�������ʼ���������󣬼�������HardExample����������  
//��ʹ��HardExampleʱ��������Ϊ0����Ϊ������������������������ά����ʼ��ʱ�õ����ֵ  
#define HardExampleNO 0 

class MySVM : public CvSVM
{
public:
	//���SVM�ľ��ߺ����е�alpha����  
	double * get_alpha_vector()
	{
		return this->decision_func->alpha;
	}

	//���SVM�ľ��ߺ����е�rho����,��ƫ����  
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
	HOGDescriptor hog(Size(100, 100), Size(20, 20), Size(10, 10), Size(10, 10), 9);//HOG���������������HOG�����ӵ�  
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������  

	Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��      
	Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ��������-1��ʾ������ 

	char * imgName = 0;
	char * temp = (char *)malloc(strlen(PosSam_PATH) + 4);

	//����������
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

		//����������
		strcpy(srcName, PosSam_PATH);
		strcat(srcName, imgName);

		Mat src = imread(srcName, 0);//��ȡͼƬ  
		if (IMAGE_SCALING)
			resize(src, src, Size(100, 100));
		cout << srcName << endl;

		vector<float> descriptors;//HOG����������  
		hog.compute(src, descriptors, Size(10, 10));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)  
		//cout<<"������ά����"<<descriptors.size()<<endl;  

		//�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������  
		if (0 == numPos)
		{
			DescriptorDim = descriptors.size();//HOG�����ӵ�ά��  
			//��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat  
			sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
			//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ��������0��ʾ������
			sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
		}

		//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat  
		for (int i = 0; i<DescriptorDim; i++)
			sampleFeatureMat.at<float>(numPos, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��  
		sampleLabelMat.at<float>(numPos, 0) = 1;//���������Ϊ1

		numPos++;
	}
	_findclose(handle);


	//��������
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

		//��������
		strcpy(srcName, NegSam_PATH);
		strcat(srcName, imgName);

		Mat src = imread(srcName, 0);//��ȡͼƬ  
		if (IMAGE_SCALING)
			resize(src, src, Size(100, 100));
		cout << srcName << endl;

		vector<float> descriptors;//HOG����������  
		hog.compute(src, descriptors, Size(10, 10));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)  
		//cout<<"������ά����"<<descriptors.size()<<endl;  

		//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat  
		for (int i = 0; i<DescriptorDim; i++)
			sampleFeatureMat.at<float>(numNeg + PosSamNO, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��  
		sampleLabelMat.at<float>(numNeg + PosSamNO, 0) = -1;//���������Ϊ0

		numNeg++;
	}
	_findclose(handle);

	//ѵ��SVM������  
	//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����  
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01  
	CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
	cout << "Training SVM classifier..." << endl;
	svm->train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//ѵ��������  
	cout << "Completed!" << endl;
	svm->save("SVM_HOG.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�  


	Mat testImg = imread("C:\\Users\\zhwei\\Desktop\\Diablo\\images\\test\\test_15.jpg");
	if (IMAGE_SCALING)
		resize(testImg, testImg, Size(100, 100));
	vector<float> descriptor;
	hog.compute(testImg, descriptor, Size(10, 10));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)  
	Mat testFeatureMat = Mat::zeros(1, descriptor.size(), CV_32FC1);//����������������������  
	//������õ�HOG�����Ӹ��Ƶ�testFeatureMat������  
	for (int i = 0; i<descriptor.size(); i++)
		testFeatureMat.at<float>(0, i) = descriptor[i];

	//��ѵ���õ�SVM�������Բ���ͼƬ�������������з���  
	int result = svm->predict(testFeatureMat);//�������  
	cout << "��������" << result << endl;

}

int setHog(HOGDescriptor * myHOG, MySVM * svm)
{
	/*************************************************************************************************
	����SVMѵ����ɺ�õ���XML�ļ����棬��һ�����飬����support vector������һ�����飬����alpha,��һ��������������rho;
	��alpha����ͬsupport vector��ˣ�ע�⣬alpha*supportVector,���õ�һ����������֮���ٸ���������������һ��Ԫ��rho��
	��ˣ���õ���һ�������������ø÷�������ֱ���滻opencv�����˼��Ĭ�ϵ��Ǹ���������cv::HOGDescriptor::setSVMDetector()����
	�Ϳ����������ѵ������ѵ�������ķ������������˼���ˡ�
	***************************************************************************************************/
	int DescriptorDim = svm->get_var_count();//����������ά������HOG�����ӵ�ά��  
	int supportVectorNum = svm->get_support_vector_count();//֧�������ĸ��� 
	cout << DescriptorDim << endl;
	cout << "֧������������" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������  
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������  
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ��  

	//��֧�����������ݸ��Ƶ�supportVectorMat������  
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm->get_support_vector(i);//���ص�i��֧������������ָ��  
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";  
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//��alpha���������ݸ��Ƶ�alphaMat��  
	double * pAlphaData = svm->get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����  
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat��  
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�  
	resultMat = -1 * alphaMat * supportVectorMat;

	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����  
	vector<float> myDetector;
	//��resultMat�е����ݸ��Ƶ�����myDetector��  
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//������ƫ����rho���õ������  
	myDetector.push_back(svm->get_rho());
	cout << "�����ά����" << myDetector.size() << endl;
	//����HOGDescriptor�ļ����  
	
	myHOG->setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());  

	//�������Ӳ������ļ�  
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
	vector<Rect> found, found_filtered;//���ο�����  
	cout << "���ж�߶ȼ��" << endl;
	myHOG->detectMultiScale(src, found, 0, Size(20, 20), Size(0, 0), 1.05, 2);//��ͼƬ���ж�߶ȼ��  
	cout << "����Ŀ�������" << found.size() << endl;

	//�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��  
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

	//�����ο�
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
	waitKey();//ע�⣺imshow֮������waitKey�������޷���ʾͼ��  


	/******************���뵥��64*128�Ĳ���ͼ������HOG�����ӽ��з���*********************/
	////��ȡ����ͼƬ(64*128��С)����������HOG������  
	////Mat testImg = imread("person014142.jpg");  
	//Mat testImg = imread("noperson000026.jpg");  
	//vector<float> descriptor;  
	//hog.compute(testImg,descriptor,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)  
	//Mat testFeatureMat = Mat::zeros(1,3780,CV_32FC1);//����������������������  
	////������õ�HOG�����Ӹ��Ƶ�testFeatureMat������  
	//for(int i=0; i<descriptor.size(); i++)  
	//  testFeatureMat.at<float>(0,i) = descriptor[i];  

	////��ѵ���õ�SVM�������Բ���ͼƬ�������������з���  
	//int result = svm.predict(testFeatureMat);//�������  
	//cout<<"��������"<<result<<endl;  

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
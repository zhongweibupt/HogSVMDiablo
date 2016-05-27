# HogSVMDiablo

##应用场景

识别图像中的扑克牌，输出扑克牌的花色和点数。

##实现方法

分为两个部分，第一步是使用Hog+SVM进行目标检测：

![Git Bash](./images/HOG+SVM.jpg)

正负样本各有100个，所以造成了误检测，增大样本容量会解决该问题。

第二步使用SIFT特征点匹配花色和点数，并输出：

![Git Bash](./images/Result.jpg)



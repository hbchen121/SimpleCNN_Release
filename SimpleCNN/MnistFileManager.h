/***********************************************
 * File: MnistFileManager.h
 *
 * Author: LYG
 * Date: 2021-04-06
 *
 * Purpose:
 *
 *
 **********************************************/

#pragma once

#include <stdio.h>
#include <xstring>
#include "Common.h"

 /*
	 简单描述一下数据格式。描述来自网络。四个文件，两个图片文件，两个标签文件。
	 标签文件：
	 1，前4个字节（第0～3个字节）是魔数2049（int型，0x00000801, 大端）;

	 2，再往后4个字节（第4～7个字节）是标签的个数：60000或10000；

	 3，再往后每1个字节是一个无符号型的数，值为0～9。

	 图片文件：
	 1，前4个字节（第0～3个字节）是魔数2051（int型，0x00000803, 大端）;

	 2，再往后4个字节（第4～7个字节）是图像的个数：60000或10000（第1个维度）；

	 3，再往后4个字节（第8～11个字节）是图像在高度上由多少个像素组成（第2个维度，高28个像素）；

	 4，再往后4个字节（第12～15个字节）是图像在宽度上由多少个像素组成（第3个维度，宽28个像素）；

	 5，再往后是一个三维数组，表示10000个或60000个分辨率28x28的灰度图像，一句话来说就是10000x28x28个像素，每个像素的值为0～255（0是背景，为白色；255是黑色）。
 */

class MnistFileManager
{
	SINGLETON(MnistFileManager)
public:
	// 加载到内存。由于训练跟测试可能是分开的，所以上层自己根据需要调用会比较合理
	bool LoadTrainDataToMemory();
	void ReleaseTrainData();
	bool LoadTestDataToMemory();
	void ReleaseTestData();
	// 随机获取，用于训练
	void GetImageAndLabelDataInRandom(int BatchCount, unsigned char*& ImageData, unsigned char*& LabelData);
	void GetImageAndLabelDataByIndex(int Index, unsigned char*& ImageData, unsigned char*& LabelData);
	// 随机获取感觉可能会导致训练波动？例如运气不大好，一直随机到差不多的位置，然后会导致loss快速下降，然后随机到了其他地方，loss会上涨？猜的
	// 所以，这里增加一个顺序训练
	// 顺序获取，用于训练或者测试，这里不需要传入batch的原因在于，获取到的是一大块内存的指针，你后续要读取多少就读取多少batch，不会有什么影响。除非index到头了才会导致越界
	void GetTestImageAndLabelDataByIndex(int Index, unsigned char*& ImageData, unsigned char*& LabelData);

	int GetTrainDataCount() const;
	int GetTestDataCount() const;
	int GetImageWidth() const;
	int GetImageHeight() const;

	std::string GetApplicationPath();
protected:
	MnistFileManager();
	~MnistFileManager();

	void TransformLabel(unsigned char* OriginalData, unsigned char* OutData, int LabelCount);
private:
	std::string mApplicationPath;

	int mTrainDataCount;
	int mTestDataCount;
	int mImageWidth;
	int mImageHeight;

	unsigned char* mImageTrainData;
	unsigned char* mImageTestData;
	unsigned char* mImageTrainLabelData;
	unsigned char* mImageTestLabelData;
};
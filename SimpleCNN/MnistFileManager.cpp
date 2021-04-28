/***********************************************
 * File: MnistFileManager.cpp
 *
 * Author: CHB
 * Date: 2021-04-07
 *
 * Purpose:
 *
 *
 **********************************************/

#include <windows.h>
#include "MnistFileManager.h"

#define TRAIN_IMAGE_FILENAME				"train-images.idx3-ubyte"
#define TRAIN_LABEL_FILENAME				"train-labels.idx1-ubyte"
#define TEST_IMAGE_FILENAME					"t10k-images.idx3-ubyte"
#define TEST_LABEL_FILENAME					"t10k-labels.idx1-ubyte"


int Big_endianToLittle_endian(int x)
{
	char Temp[4];

	Temp[0] = x >> 24;
	Temp[1] = x << 8 >> 24;
	Temp[2] = x << 16 >> 24;
	Temp[3] = x << 24 >> 24;

	int Result = 0;

	memcpy(&Result, Temp, 4);
	return Result;
}

SINGLETON_IMPL(MnistFileManager)

MnistFileManager::MnistFileManager()
{
	mTrainDataCount = 0;
	mTestDataCount = 0;
	mImageWidth = 0;
	mImageHeight = 0;

	mImageTrainData = nullptr;
	mImageTestData = nullptr;
	mImageTrainLabelData = nullptr;
	mImageTestLabelData = nullptr;
}

MnistFileManager::~MnistFileManager()
{
	ReleaseTrainData();
	ReleaseTestData();
}

bool MnistFileManager::LoadTrainDataToMemory()
{
	if (mImageTrainData && mImageTrainLabelData) return true;

	std::string FullPath = GetApplicationPath();

	std::string FileName;
	FileName = FullPath + TRAIN_IMAGE_FILENAME;
	FILE* fp = nullptr;
	fopen_s(&fp, FileName.c_str(), "rb");
	if (fp == nullptr) {
		printf("Please Put Mnist datasets in %s\n", FullPath.c_str());
		return false;
	}
	fseek(fp, 4, SEEK_SET);
	fread_s(&mTrainDataCount, sizeof(mTrainDataCount), sizeof(mTrainDataCount), 1, fp);
	fread_s(&mImageHeight, sizeof(mImageHeight), sizeof(mImageHeight), 1, fp);
	fread_s(&mImageWidth, sizeof(mImageWidth), sizeof(mImageWidth), 1, fp);

	mTrainDataCount = Big_endianToLittle_endian(mTrainDataCount);
	mImageHeight = Big_endianToLittle_endian(mImageHeight);
	mImageWidth = Big_endianToLittle_endian(mImageWidth);

	int ImageDataSize = mTrainDataCount * mImageHeight * mImageWidth;
	mImageTrainData = new unsigned char[ImageDataSize];
	fread_s(mImageTrainData, ImageDataSize, ImageDataSize, 1, fp);

	fclose(fp);

	// 读取Label数据
	FileName = FullPath + TRAIN_LABEL_FILENAME;
	fopen_s(&fp, FileName.c_str(), "rb");
	if (fp == nullptr) return false;

	fseek(fp, 8, SEEK_SET);

	// 读取出来文件的数据
	unsigned char* Temp = new unsigned char[mTrainDataCount];
	// 10的分类，把原本的0、1、2、3、……等数据转换成分类数据
	mImageTrainLabelData = new unsigned char[mTrainDataCount * 10];
	fread_s(Temp, mTrainDataCount, mTrainDataCount, 1, fp);
	fclose(fp);

	TransformLabel(Temp, mImageTrainLabelData, mTrainDataCount);

	SAFE_DELETE_ARRAY(Temp);

	return true;
}

void MnistFileManager::ReleaseTrainData()
{
	SAFE_DELETE_ARRAY(mImageTrainData);
	SAFE_DELETE_ARRAY(mImageTrainLabelData);
}

bool MnistFileManager::LoadTestDataToMemory()
{
	if (mImageTestData && mImageTestLabelData) return true;

	std::string FullPath = GetApplicationPath();

	std::string FileName;
	FileName = FullPath + TEST_IMAGE_FILENAME;
	FILE* fp = nullptr;
	fopen_s(&fp, FileName.c_str(), "rb");
	if (fp == nullptr) return false;

	fseek(fp, 4, SEEK_SET);
	fread_s(&mTestDataCount, sizeof(mTestDataCount), sizeof(mTestDataCount), 1, fp);
	fread_s(&mImageHeight, sizeof(mImageHeight), sizeof(mImageHeight), 1, fp);
	fread_s(&mImageWidth, sizeof(mImageWidth), sizeof(mImageWidth), 1, fp);

	mTestDataCount = Big_endianToLittle_endian(mTestDataCount);
	mImageHeight = Big_endianToLittle_endian(mImageHeight);
	mImageWidth = Big_endianToLittle_endian(mImageWidth);

	int ImageDataSize = mTestDataCount * mImageHeight * mImageWidth;
	mImageTestData = new unsigned char[ImageDataSize];
	fread_s(mImageTestData, ImageDataSize, ImageDataSize, 1, fp);

	fclose(fp);

	// 读取Label数据
	FileName = FullPath + TEST_LABEL_FILENAME;
	fopen_s(&fp, FileName.c_str(), "rb");
	if (fp == nullptr) return false;

	fseek(fp, 8, SEEK_SET);

	// 读取出来文件的数据
	unsigned char* Temp = new unsigned char[mTestDataCount];
	// 10的分类，把原本的0、1、2、3、……等数据转换成分类数据
	mImageTestLabelData = new unsigned char[mTestDataCount * 10];
	fread_s(Temp, mTestDataCount, mTestDataCount, 1, fp);
	fclose(fp);

	TransformLabel(mImageTestLabelData, Temp, mTestDataCount);

	SAFE_DELETE_ARRAY(Temp);

	return true;
}

void MnistFileManager::ReleaseTestData()
{
	SAFE_DELETE_ARRAY(mImageTestData);
	SAFE_DELETE_ARRAY(mImageTestLabelData);
}

void MnistFileManager::GetImageAndLabelDataInRandom(int BatchCount, unsigned char*& ImageData, unsigned char*& LabelData)
{
	int MaxRange = mTrainDataCount - BatchCount - 1;

	int Index = rand() % MaxRange;

	// 写死会快一点点:(
	const int ImageSize = 28 * 28;

	ImageData = mImageTrainData + Index * ImageSize;
	LabelData = mImageTrainLabelData + Index * 10;
}

void MnistFileManager::GetImageAndLabelDataByIndex(int Index, unsigned char*& ImageData, unsigned char*& LabelData)
{
	// 写死会快一点点:(
	const int ImageSize = 28 * 28;

	ImageData = mImageTrainData + Index * ImageSize;
	LabelData = mImageTrainLabelData + Index * 10;
}

void MnistFileManager::GetTestImageAndLabelDataByIndex(int Index, unsigned char*& ImageData, unsigned char*& LabelData)
{
	// 写死会快一点点:(
	const int ImageSize = 28 * 28;

	ImageData = mImageTestData + Index * ImageSize;
	LabelData = mImageTestLabelData + Index * 10;
}

int MnistFileManager::GetTrainDataCount() const
{
	return mTrainDataCount;
}
int MnistFileManager::GetTestDataCount() const
{
	return mTestDataCount;
}
int MnistFileManager::GetImageWidth() const
{
	return mImageWidth;
}
int MnistFileManager::GetImageHeight() const
{
	return mImageHeight;
}

std::string MnistFileManager::GetApplicationPath()
{
	if (mApplicationPath == "")
	{
		char Temp[256];
		memset(Temp, 0, 256);

		GetModuleFileNameA(NULL, Temp, 256);
		int nLen = (int)strlen(Temp);
		while (nLen)
		{
			if (Temp[nLen] == '\\' || Temp[nLen] == '/')
			{
				break;
			}
			Temp[nLen--] = '\0';
		}
		mApplicationPath = Temp;
	}
	return mApplicationPath;
}

void MnistFileManager::TransformLabel(unsigned char* OriginalData, unsigned char* OutData, int LabelCount)
{
	//把标签数据转换成softmax类似的分类数据
	unsigned char TempMap[10][10] = { { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },{ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 } ,{ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 } ,{ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 } ,{ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 } ,
	{ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 } ,{ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 } ,{ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 } ,{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 } ,{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 } };

	int Index = 0;
	for (int i = 0; i < LabelCount; i++)
	{
		int Index = OriginalData[i];
		memcpy(OutData + i * 10, TempMap[Index], 10);
	}
}
/***********************************************
 * File: SdlfCalculatorCPU.cpp
 *
 * Author: CHB
 * Date: 2020-04-08
 *
 * Purpose:
 *
 *
 **********************************************/

#include "SdlfCalculatorCPU.h"
#include "SdlfFunction.h"
#include <string.h>
#include <algorithm>

SdlfCalculatorCPU::SdlfCalculatorCPU()
{
	mImageOutData = nullptr;
}

SdlfCalculatorCPU::~SdlfCalculatorCPU()
{
	SAFE_DELETE_ARRAY(mImageOutData);
}

void SdlfCalculatorCPU::Transform_uchar_to_float(unsigned char* InData, float* OutData, int DataLen)
{
	for (int i = 0; i < DataLen; i++) {
		OutData[i] = UCharToFloat(InData[i]);
	}
}

void SdlfCalculatorCPU::Conv2D(float* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel* CK, SdlfActivationFunc ActivationFunc)
{
	mInWidth = mOutWidth = ImageWidth;
	mInHeight = mOutHeight = ImageHeight;
	mInChannel = ImageChannel;

	mOutChannel = CK->ConvKernelCount;
	mBatchSize = BatchSize;
	mPadding = (CK->ConvKernelWidth - 1) >> 1;

	// 输出数据
	// 这里要乘以CK->ConvKernelCount，一张图其实是分开了CK->ConvKernelCount那么多张特征图
	SAFE_DELETE_ARRAY(mImageOutData);
	mImageOutData = new float[BatchSize * ImageWidth * ImageHeight * CK->ConvKernelCount];

	memset(mImageOutData, 0, sizeof(float) * BatchSize * ImageWidth * ImageHeight * CK->ConvKernelCount);

	float* ReluGradientData = new float[mInWidth * mInHeight * BatchSize * CK->ConvKernelCount];

	memset(ReluGradientData, 0, sizeof(float) * mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);

	// 开始计算卷积
	if (CK->CB == nullptr)
	{
		CK->CB = new ConvBlocks;
	}
	// 为 CB 分配内存
	SdlfFunction::BuildConvArray(CK->CB, ImageData, ImageWidth, ImageHeight, ImageChannel, CK->ConvKernelWidth, CK->ConvKernelHeight, BatchSize);
	// 每个batch可以分成这么多个小块
	int ConvCountPerBatch = mInWidth * mInHeight;
	SdlfFunction::Conv2D(CK->CB, ConvCountPerBatch, CK, mImageOutData, ConvCountPerBatch, ReluGradientData, BatchSize);

	CK->ApplyImageReluGradientData(ReluGradientData, mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);
	CK->ApplyImageInputWidthAndHeight(mInWidth, mInHeight);
	SAFE_DELETE_ARRAY(ReluGradientData);
}

void SdlfCalculatorCPU::Conv2D(ConvKernel* CK, SdlfActivationFunc ActivationFunc)
{
	// 以上一次的输出作为这次的输入，进行卷积计算
	mInWidth = mOutWidth;
	mInHeight = mOutHeight;
	mInChannel = CK->ConvKernelChannel;

	// 输出channel是卷积核的channel
	mOutChannel = CK->ConvKernelCount;
	mPadding = (CK->ConvKernelWidth - 1) >> 1;

	// 这里要乘以CK->ConvKernelCount，一张图其实是分开了CK->ConvKernelCount那么多张特征图
	float* OutData = new float[mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount];
	float* ReluGradientData = new float[mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount];

	memset(OutData, 0, sizeof(float) * mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);
	memset(ReluGradientData, 0, sizeof(float) * mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);

	// 开始做卷积计算
	if (CK->CB == nullptr)
	{
		CK->CB = new ConvBlocks;
	}
	SdlfFunction::BuildConvArray(CK->CB, mImageOutData, mInWidth, mInHeight, mInChannel, CK->ConvKernelWidth, CK->ConvKernelHeight, mBatchSize);
	// 每个batch可以分成这么多个小块
	int ConvCountPerBatch = mInWidth * mInHeight;
	SdlfFunction::Conv2D(CK->CB, ConvCountPerBatch, CK, OutData, ConvCountPerBatch, ReluGradientData, mBatchSize);

	// CK->ApplyImageReluGradientData(ReluGradientData, mInWidth * mInHeight * mBatchCount * CK->ConvKernelCount);
	CK->ApplyImageInputWidthAndHeight(mInWidth, mInHeight);
	SAFE_DELETE_ARRAY(ReluGradientData);
	SAFE_DELETE_ARRAY(mImageOutData);
	mImageOutData = OutData;
}

void SdlfCalculatorCPU::Max_Pool_2_2(int InWidth, int InHeight, float* InData, float* MaxPoolGradientData, float* OutData)
{
	float* PosOut;
	float Temp[4];
	for (int i = 0; i < InHeight / 2; i++)
	{
		for (int j = 0; j < InWidth / 2; j++)
		{
			int P[4] = { (i * 2 * InWidth + j * 2), (i * 2 * InWidth + j * 2 + 1), ((i * 2 + 1) * InWidth + j * 2) , ((i * 2 + 1) * InWidth + j * 2 + 1) };
			PosOut = &(OutData[(i * InWidth / 2 + j)]);

			Temp[0] = InData[P[0]];
			Temp[1] = InData[P[1]];
			Temp[2] = InData[P[2]];
			Temp[3] = InData[P[3]];
			// 有可能四个值相等，这个时候，用均值池化
			if (Temp[0] == Temp[1] && Temp[0] == Temp[2] && Temp[0] == Temp[3])
			{
				PosOut[0] = Temp[0];
				MaxPoolGradientData[P[0]] = 0.25f;
				MaxPoolGradientData[P[1]] = 0.25f;
				MaxPoolGradientData[P[2]] = 0.25f;
				MaxPoolGradientData[P[3]] = 0.25f;
			}
			else
			{
				int MaxIndex = 0;
				PosOut[0] = SdlfFunction::MaxInPool(Temp, 4, MaxIndex);
				MaxPoolGradientData[P[MaxIndex]] = 1.0f;
			}
		}
	}
	return;
}

void SdlfCalculatorCPU::Max_Pool_2_2(ConvKernel* CK)
{
	int OriginalImageSize = mInWidth * mInHeight;
	// 除以4是因为池化输出本来就是宽高一边除以2
	int ImageSize = OriginalImageSize >> 2;
	float* OutData = new float[ImageSize * CK->ConvKernelCount * mBatchSize];
	float* MaxPoolGradientData = new float[OriginalImageSize * CK->ConvKernelCount * mBatchSize];
	memset(MaxPoolGradientData, 0, sizeof(float) * OriginalImageSize * CK->ConvKernelCount * mBatchSize);

	for (int i = 0; i < mBatchSize * CK->ConvKernelCount; i++)
	{
		int InPos = i * OriginalImageSize;
		int OutPos = i * ImageSize;
		Max_Pool_2_2(mInWidth, mInHeight, mImageOutData + InPos, MaxPoolGradientData + InPos, OutData + OutPos);
	}

	CK->ApplyImageMaxPoolGradientData(MaxPoolGradientData, OriginalImageSize * CK->ConvKernelCount * mBatchSize);

	SAFE_DELETE_ARRAY(mImageOutData);
	SAFE_DELETE_ARRAY(MaxPoolGradientData);
	mImageOutData = OutData;
	mOutWidth = mInWidth >> 1;
	mOutHeight = mInHeight >> 1;
	mOutChannel = CK->ConvKernelCount;
}

void SdlfCalculatorCPU::CalcFullLink(float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, ConvKernel* FLinkKernel)
{
	float* OutData = new float[mBatchSize * FLinkKernel->ConvKernelCount];
	float* DropOutGradient = new float[mBatchSize * FLinkKernel->ConvKernelCount];
	float* ReluGradient = new float[mBatchSize * FLinkKernel->ConvKernelCount];

	memset(OutData, 0, sizeof(float) * mBatchSize * FLinkKernel->ConvKernelCount);
	memset(DropOutGradient, 0, sizeof(float) * mBatchSize * FLinkKernel->ConvKernelCount);
	memset(ReluGradient, 0, sizeof(float) * mBatchSize * FLinkKernel->ConvKernelCount);

	// 全连接之后，输出的图片宽度跟高度之类的，全部没有了，应该是只有一个channel了。
	mOutWidth = ImageWidth;
	mOutHeight = ImageHeight;
	mOutChannel = ImageDepth;
	mInWidth = mOutWidth;
	mInHeight = mOutHeight;
	mInChannel = mOutChannel;

	if (FLinkKernel->CB == nullptr)
	{
		FLinkKernel->CB = new ConvBlocks;
	}
	SdlfFunction::BuildFullConnectedArray(FLinkKernel->CB, ImageData, ImageWidth, ImageHeight, ImageDepth, mBatchSize);

	SdlfFunction::ConvFullConnected(FLinkKernel->CB, FLinkKernel, OutData, ReluGradient, mBatchSize);
	FLinkKernel->ApplyImageInputWidthAndHeight(mOutWidth, mOutHeight);

	// 这个大概是变成了1还是0了？一张图片已已经全部变成了N个数
	mOutWidth = mOutHeight = 1;
	mOutChannel = FLinkKernel->ConvKernelCount;
	SAFE_DELETE_ARRAY(mImageOutData);
	mImageOutData = OutData;

	// FLinkKernel->ApplyDropOutGradient(DropOutGradient, mBatchSize * FLinkKernel->ConvKernelCount);
	SAFE_DELETE_ARRAY(DropOutGradient);
	FLinkKernel->ApplyImageReluGradientData(ReluGradient, mBatchSize * FLinkKernel->ConvKernelCount);
	SAFE_DELETE_ARRAY(ReluGradient);
}

void SdlfCalculatorCPU::CalcFullLink(ConvKernel* FLinkKernel)
{
	float* OutData = new float[mBatchSize * FLinkKernel->ConvKernelCount];
	float* DropOutGradient = new float[mBatchSize * FLinkKernel->ConvKernelCount];
	float* ReluGradient = new float[mBatchSize * FLinkKernel->ConvKernelCount];

	memset(OutData, 0, sizeof(float) * mBatchSize * FLinkKernel->ConvKernelCount);
	memset(DropOutGradient, 0, sizeof(float) * mBatchSize * FLinkKernel->ConvKernelCount);
	memset(ReluGradient, 0, sizeof(float) * mBatchSize * FLinkKernel->ConvKernelCount);

	// 全连接之后，输出的图片宽度跟高度之类的，全部没有了，应该是只有一个channel了。
	mInWidth = mOutWidth;
	mInHeight = mOutHeight;
	mInChannel = mOutChannel;

	// apply image input data
	if (FLinkKernel->CB == nullptr)
	{
		FLinkKernel->CB = new ConvBlocks;
	}
	SdlfFunction::BuildFullConnectedArray(FLinkKernel->CB, mImageOutData, mInWidth, mInHeight, mInChannel, mBatchSize);

	SdlfFunction::ConvFullConnected(FLinkKernel->CB, FLinkKernel, OutData, ReluGradient, mBatchSize);
	FLinkKernel->ApplyImageInputWidthAndHeight(mOutWidth, mOutHeight);

	// 另外，在这里直接DropOut算了。不然还要另外折腾半天
	// SdlfFunction::DropOut(OutData, DropOutGradient, mBatchCount * FLinkKernel->ConvKernelCount, DROP_OUT_PARAM);
	// 这个大概是变成了1还是0了？一张图片已已经全部变成了N个数
	mOutWidth = mOutHeight = 1;
	mOutChannel = FLinkKernel->ConvKernelCount;
	SAFE_DELETE_ARRAY(mImageOutData);
	mImageOutData = OutData;

	// FLinkKernel->ApplyDropOutGradient(DropOutGradient, mBatchSize * FLinkKernel->ConvKernelCount);
	SAFE_DELETE_ARRAY(DropOutGradient);
	FLinkKernel->ApplyImageReluGradientData(ReluGradient, mBatchSize * FLinkKernel->ConvKernelCount);
	SAFE_DELETE_ARRAY(ReluGradient);
}


float* SdlfCalculatorCPU::SoftMax(SoftMaxKernel* SMK)
{
	// 先把1 * 1024的数据转换成1 * 10的。其实类似于矩阵乘法，乘以一个1024 * 10的矩阵。这里直接做个简单点积即可。
	float* OutData = new float[SMK->Column * mBatchSize];
	for (int i = 0; i < mBatchSize; i++)
	{
		for (int j = 0; j < SMK->Column; j++)
		{
			int Pos = i * SMK->Column + j;
			OutData[Pos] = SMK->DotProduct(&(mImageOutData[SMK->Row * i]), j);
		}
		// softmax
		int Pos = i * SMK->Column;
		float Out[10];
		//SdlfFunction::softmax_default(&OutData[Pos], Out, 10);
		SdlfFunction::softmax(&OutData[Pos], Out, 10);
		memcpy(&OutData[Pos], Out, 10 * sizeof(float));
	}
	mOutChannel = SMK->Column;
	// 完全是补救的代码
	SMK->ApplyLastInput(mImageOutData, mBatchSize * SMK->Row);
	SAFE_DELETE_ARRAY(mImageOutData);
	mImageOutData = OutData;

	return mImageOutData;
}

float* SdlfCalculatorCPU::SoftMax()
{
	float* OutData = new float[mInChannel * mBatchSize];
	for (int i = 0; i < mBatchSize; i++)
	{
		// softmax
		int Pos = i * mInChannel;
		float Out[10];
		//SdlfFunction::softmax_default(&OutData[Pos], Out, 10);
		SdlfFunction::softmax(&OutData[Pos], Out, 10);
		memcpy(&OutData[Pos], Out, 10 * sizeof(float));
	}
	mOutChannel = mInChannel;
	mImageOutData = OutData;
	return mImageOutData;
}

void SdlfCalculatorCPU::Release()
{
	delete this;
}

/***********************************************
 * File: SdlfCalculatorCUDA.cpp
 *
 * Author: CHB
 * Date: 2020-04-19
 *
 * Purpose:
 *
 *
 **********************************************/
#include "SdlfCalculatorCUDA.h"
#include "SdlfFunctionCUDA.h"
#include <string.h>
#include <algorithm>

SdlfCalculatorCUDA::SdlfCalculatorCUDA()
{
	mImageOutData = nullptr;
}

SdlfCalculatorCUDA::~SdlfCalculatorCUDA()
{
	SAFE_DELETE_ARRAY(mImageOutData);
}

void SdlfCalculatorCUDA::Transform_uchar_to_float(unsigned char* InData, float* OutData, int DataLen)
{
	for (int i = 0; i < DataLen; i++) {
		OutData[i] = UCharToFloat(InData[i]);
	}
}

extern "C" void BuildConvArrayCUDA(ConvBlocksCUDA * d_CBC, float* d_ImageData, int ImageWidth, int ImageHeight,
	int ImageDepth, int ConvWidth, int ConvHeight, int BatchSize);

extern "C" float* Conv2DCUDA_2(ConvBlocksCUDA * CBC, int ConvBlocksPerBatch, ConvKernelCUDA * CKC, float* ImageOutData,
	int ImageSize2D, float* ReluOutData, int BatchSize, int InChannel, int OutChannel);

extern "C" float* Conv2DCUDA_3(float* d_Indata, ConvKernelCUDA * d_CKC, float* ImageOutData,
	int ImageWidth, int ImageHeight, float* ReluOutData, int BatchSize,
	int InChannel, int OutChannel);


// Conv2D与Conv2D_2的区别是，Conv2D_2沿用了CPU的卷积方式，先构建CB，再计算；
// Conv2D则把他们放在了一起，略过了CB的构建，但这会造成梯度下降时参数更新的错误，因此只能前向传播
// Conv2D 就是单纯的想实现“标准的GPU卷积过程”，以加速卷积。关于卷积加速可参考文章：https://blog.csdn.net/qq_40491305/article/details/116236956
void SdlfCalculatorCUDA::Conv2D(float* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel* CK, SdlfActivationFunc ActivationFunc)
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

	int ImageSize2D = ImageWidth * ImageHeight;
	cudaError_t Err;
	int InChannel = CK->ConvKernelChannel, OutChannel = CK->ConvKernelCount;
	int InSize = mBatchSize * InChannel * ImageSize2D * sizeof(float), OutSize = mBatchSize * OutChannel * ImageSize2D * sizeof(float);
	int ConvLen = CK->ConvKernelWidth * CK->ConvKernelHeight * InChannel;
	if (CK->CB->ConvArray == nullptr) {
		CK->CB->ConvArray = new float[ImageSize2D * mBatchSize * InChannel];
		CK->CB->ArrayLen = InChannel;
	}
	ConvKernelCUDA* CKC = new ConvKernelCUDA(CK), * d_CKC;

	Err = CUDA_MALLOC(d_CKC, sizeof(ConvKernelCUDA));
	HANDLE_ERROR(Err);
	Err = cudaMemcpy(d_CKC, CKC, sizeof(ConvKernelCUDA), cudaMemcpyHostToDevice);
	HANDLE_ERROR(Err);

	float* d_InData = new float[InSize / sizeof(float)];
	Err = CUDA_MALLOC(d_InData, InSize);
	HANDLE_ERROR(Err);
	Err = cudaMemcpy(d_InData, ImageData, InSize, cudaMemcpyHostToDevice);
	HANDLE_ERROR(Err);

	// 每个batch可以分成这么多个小块
	Conv2DCUDA_3(d_InData, d_CKC, mImageOutData, ImageWidth, ImageHeight, ReluGradientData, mBatchSize,
					InChannel, OutChannel);

	HANDLE_ERROR(Err);
	Err = cudaFree(d_CKC);
	HANDLE_ERROR(Err);
	Err = cudaFree(d_InData);
	HANDLE_ERROR(Err);
	SAFE_DELETE(CKC);
	CK->ApplyImageReluGradientData(ReluGradientData, mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);
	CK->ApplyImageInputWidthAndHeight(mInWidth, mInHeight);
	SAFE_DELETE_ARRAY(ReluGradientData);
}

void SdlfCalculatorCUDA::Conv2D(ConvKernel* CK, SdlfActivationFunc ActivationFunc)
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
	int ImageSize2D = mOutWidth * mOutHeight;
	cudaError_t Err;
	int InChannel = CK->ConvKernelChannel, OutChannel = CK->ConvKernelCount;
	int InSize = mBatchSize * InChannel * ImageSize2D * sizeof(float), OutSize = mBatchSize * OutChannel * ImageSize2D * sizeof(float);
	int ConvLen = CK->ConvKernelWidth * CK->ConvKernelHeight * InChannel;
	if (CK->CB->ConvArray == nullptr) {
		CK->CB->ConvArray = new float[ImageSize2D * mBatchSize * InChannel];
		CK->CB->ArrayLen = InChannel;
	}
	ConvKernelCUDA* CKC = new ConvKernelCUDA(CK), * d_CKC;

	Err = CUDA_MALLOC(d_CKC, sizeof(ConvKernelCUDA));
	HANDLE_ERROR(Err);
	Err = cudaMemcpy(d_CKC, CKC, sizeof(ConvKernelCUDA), cudaMemcpyHostToDevice);
	HANDLE_ERROR(Err);

	float* d_InData = new float[InSize / sizeof(float)];
	Err = CUDA_MALLOC(d_InData, InSize);
	HANDLE_ERROR(Err);
	Err = cudaMemcpy(d_InData, mImageOutData, InSize, cudaMemcpyHostToDevice);
	HANDLE_ERROR(Err);

	// 每个batch可以分成这么多个小块
	Conv2DCUDA_3(d_InData, d_CKC, mImageOutData, mOutWidth, mOutHeight, ReluGradientData, mBatchSize,
		InChannel, OutChannel);

	HANDLE_ERROR(Err);
	Err = cudaFree(d_CKC);
	HANDLE_ERROR(Err);
	Err = cudaFree(d_InData);
	HANDLE_ERROR(Err);
	SAFE_DELETE(CKC);
	//SdlfFunctionCUDA::BuildConvArray(CK->CB, mImageOutData, mInWidth, mInHeight, mInChannel, CK->ConvKernelWidth, CK->ConvKernelHeight, mBatchSize);
	//// 每个batch可以分成这么多个小块
	//int ConvCountPerBatch = mInWidth * mInHeight;
	//Conv2DCUDA(CK->CB, ConvCountPerBatch, CK, mImageOutData, ConvCountPerBatch, ReluGradientData, mBatchSize);
	//// SdlfFunctionCUDA::Conv2D(CK->CB, ConvCountPerBatch, CK, OutData, ConvCountPerBatch, ReluGradientData, mBatchSize);

	//// CK->ApplyImageReluGradientData(ReluGradientData, mInWidth * mInHeight * mBatchCount * CK->ConvKernelCount);
	CK->ApplyImageInputWidthAndHeight(mInWidth, mInHeight);
	SAFE_DELETE_ARRAY(ReluGradientData);
	SAFE_DELETE_ARRAY(mImageOutData);
	mImageOutData = OutData;
}

void SdlfCalculatorCUDA::Conv2D_2(float* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel* CK, SdlfActivationFunc ActivationFunc)
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
	int ImageSize2D = ImageWidth * ImageHeight;
	cudaError_t Err;
	int InChannel = CK->ConvKernelChannel, OutChannel = CK->ConvKernelCount;
	int InSize = mBatchSize * InChannel * ImageSize2D * sizeof(float), OutSize = mBatchSize * OutChannel * ImageSize2D * sizeof(float);
	int ConvLen = CK->ConvKernelWidth * CK->ConvKernelHeight * InChannel;
	if (CK->CB->ConvArray == nullptr) {
		CK->CB->ConvArray = new float[ImageSize2D * mBatchSize * ConvLen];
		CK->CB->ArrayLen = ConvLen;
	}

	ConvBlocksCUDA* CBC = new ConvBlocksCUDA(CK->CB, mBatchSize, ImageSize2D), * d_CBC;
	ConvKernelCUDA* CKC = new ConvKernelCUDA(CK), * d_CKC;

	Err = CUDA_MALLOC(d_CKC, sizeof(ConvKernelCUDA));
	HANDLE_ERROR(Err);
	Err = cudaMemcpy(d_CKC, CKC, sizeof(ConvKernelCUDA), cudaMemcpyHostToDevice);
	HANDLE_ERROR(Err);
	Err = CUDA_MALLOC(d_CBC, sizeof(ConvBlocksCUDA));
	HANDLE_ERROR(Err);
	Err = cudaMemcpy(d_CBC, CBC, sizeof(ConvBlocksCUDA), cudaMemcpyHostToDevice);
	HANDLE_ERROR(Err);

	Err = cudaMemset(CBC->ConvArray, 0, ImageSize2D * mBatchSize * ConvLen * sizeof(float));
	HANDLE_ERROR(Err);
	float* d_InData = new float[InSize / sizeof(float)];
	Err = CUDA_MALLOC(d_InData, InSize);
	HANDLE_ERROR(Err);
	Err = cudaMemcpy(d_InData, ImageData, InSize, cudaMemcpyHostToDevice);
	HANDLE_ERROR(Err);

	// 为 CB 分配内存
	BuildConvArrayCUDA(d_CBC, d_InData, mInWidth, mInHeight, mInChannel, CK->ConvKernelWidth,
		CK->ConvKernelHeight, mBatchSize);
	// 每个batch可以分成这么多个小块
	Conv2DCUDA_2(d_CBC, ImageSize2D, d_CKC, mImageOutData, ImageSize2D, ReluGradientData, mBatchSize,
		InChannel, OutChannel);
	//SdlfFunctionCUDA::Conv2D(CK->CB, ConvCountPerBatch, CK, mImageOutData, ConvCountPerBatch, ReluGradientData, BatchSize);
	Err = cudaMemcpy(CBC, d_CBC, sizeof(ConvBlocksCUDA), cudaMemcpyDeviceToHost);
	HANDLE_ERROR(Err);
	Err = cudaFree(d_CBC);
	HANDLE_ERROR(Err);
	Err = cudaFree(d_CKC);
	HANDLE_ERROR(Err);
	Err = cudaFree(d_InData);
	HANDLE_ERROR(Err);
	SAFE_DELETE(CKC);
	SAFE_DELETE(CBC)
		CK->ApplyImageReluGradientData(ReluGradientData, mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);
	CK->ApplyImageInputWidthAndHeight(mInWidth, mInHeight);
	SAFE_DELETE_ARRAY(ReluGradientData);
}

void SdlfCalculatorCUDA::Conv2D_2(ConvKernel* CK, SdlfActivationFunc ActivationFunc)
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
	int ImageSize2D = mInWidth * mInHeight;
	cudaError_t Err;
	int InChannel = CK->ConvKernelChannel, OutChannel = CK->ConvKernelCount;
	int InSize = mBatchSize * InChannel * ImageSize2D * sizeof(float), OutSize = mBatchSize * OutChannel * ImageSize2D * sizeof(float);
	int ConvLen = CK->ConvKernelWidth * CK->ConvKernelHeight * InChannel;
	if (CK->CB->ConvArray == nullptr) {
		CK->CB->ConvArray = new float[ImageSize2D * mBatchSize * ConvLen];
		CK->CB->ArrayLen = ConvLen;
	}

	ConvBlocksCUDA* CBC = new ConvBlocksCUDA(CK->CB, mBatchSize, ImageSize2D), * d_CBC;
	ConvKernelCUDA* CKC = new ConvKernelCUDA(CK), * d_CKC;

	Err = CUDA_MALLOC(d_CKC, sizeof(ConvKernelCUDA));
	HANDLE_ERROR(Err);
	Err = cudaMemcpy(d_CKC, CKC, sizeof(ConvKernelCUDA), cudaMemcpyHostToDevice);
	HANDLE_ERROR(Err);
	Err = CUDA_MALLOC(d_CBC, sizeof(ConvBlocksCUDA));
	HANDLE_ERROR(Err);
	Err = cudaMemcpy(d_CBC, CBC, sizeof(ConvBlocksCUDA), cudaMemcpyHostToDevice);
	HANDLE_ERROR(Err);

	float* d_InData = new float[InSize / sizeof(float)];
	Err = CUDA_MALLOC(d_InData, InSize);
	HANDLE_ERROR(Err);
	Err = cudaMemcpy(d_InData, mImageOutData, InSize, cudaMemcpyHostToDevice);
	HANDLE_ERROR(Err);

	// 为 d_CBC 分配内存
	BuildConvArrayCUDA(d_CBC, d_InData, mInWidth, mInHeight, mInChannel, CK->ConvKernelWidth,
		CK->ConvKernelHeight, mBatchSize);
	// 每个batch可以分成这么多个小块

	Conv2DCUDA_2(d_CBC, ImageSize2D, d_CKC, mImageOutData, ImageSize2D, ReluGradientData, mBatchSize,
		InChannel, OutChannel);
	//SdlfFunctionCUDA::Conv2D(CK->CB, ConvCountPerBatch, CK, mImageOutData, ConvCountPerBatch, ReluGradientData, BatchSize);
	Err = cudaMemcpy(CBC, d_CBC, sizeof(ConvBlocksCUDA), cudaMemcpyDeviceToHost);
	HANDLE_ERROR(Err);
	Err = cudaFree(d_CBC);
	HANDLE_ERROR(Err);
	Err = cudaFree(d_CKC);
	HANDLE_ERROR(Err);
	Err = cudaFree(d_InData);
	HANDLE_ERROR(Err);
	SAFE_DELETE(CKC);
	SAFE_DELETE(CBC)
		//SdlfFunctionCUDA::BuildConvArray(CK->CB, mImageOutData, mInWidth, mInHeight, mInChannel, CK->ConvKernelWidth, CK->ConvKernelHeight, mBatchSize);
		//// 每个batch可以分成这么多个小块
		//int ConvCountPerBatch = mInWidth * mInHeight;
		//Conv2DCUDA(CK->CB, ConvCountPerBatch, CK, mImageOutData, ConvCountPerBatch, ReluGradientData, mBatchSize);
		//// SdlfFunctionCUDA::Conv2D(CK->CB, ConvCountPerBatch, CK, OutData, ConvCountPerBatch, ReluGradientData, mBatchSize);

		//// CK->ApplyImageReluGradientData(ReluGradientData, mInWidth * mInHeight * mBatchCount * CK->ConvKernelCount);
		CK->ApplyImageInputWidthAndHeight(mInWidth, mInHeight);
	SAFE_DELETE_ARRAY(ReluGradientData);
	SAFE_DELETE_ARRAY(mImageOutData);
	mImageOutData = OutData;
}

void SdlfCalculatorCUDA::Max_Pool_2_2(int InWidth, int InHeight, float* InData, float* MaxPoolGradientData, float* OutData)
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
				PosOut[0] = SdlfFunctionCUDA::MaxInPool(Temp, 4, MaxIndex);
				MaxPoolGradientData[P[MaxIndex]] = 1.0f;
			}
		}
	}
	return;
}

extern "C" void Max_Pool_2_2_CUDA(ConvKernel* CK, int BatchSize, int ImgWidth, int ImgHeight, int Depth, float* InData, float* OutData, float* GradData);

void SdlfCalculatorCUDA::Max_Pool_2_2(ConvKernel* CK)
{
	int OriginalImageSize = mInWidth * mInHeight;
	// 除以4是因为池化输出本来就是宽高一边除以2
	int ImageSize = OriginalImageSize >> 2;
	float* OutData = new float[ImageSize * CK->ConvKernelCount * mBatchSize];
	float* MaxPoolGradientData = new float[OriginalImageSize * CK->ConvKernelCount * mBatchSize];
	memset(MaxPoolGradientData, 0, sizeof(float) * OriginalImageSize * CK->ConvKernelCount * mBatchSize);

	// __CUDA__
	Max_Pool_2_2_CUDA(CK, mBatchSize, mInWidth, mInHeight, CK->ConvKernelCount, mImageOutData, OutData, MaxPoolGradientData);
	/*for (int i = 0; i < mBatchSize * CK->ConvKernelCount; i++)
	{
		int InPos = i * OriginalImageSize;
		int OutPos = i * ImageSize;
		Max_Pool_2_2(mInWidth, mInHeight, mImageOutData + InPos, MaxPoolGradientData + InPos, OutData + OutPos);
	}*/

	CK->ApplyImageMaxPoolGradientData(MaxPoolGradientData, OriginalImageSize * CK->ConvKernelCount * mBatchSize);

	SAFE_DELETE_ARRAY(mImageOutData);
	SAFE_DELETE_ARRAY(MaxPoolGradientData);
	mImageOutData = OutData;
	mOutWidth = mInWidth >> 1;
	mOutHeight = mInHeight >> 1;
	mOutChannel = CK->ConvKernelCount;
}

void SdlfCalculatorCUDA::CalcFullLink(float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, ConvKernel* FLinkKernel)
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
	// 把 imgData 内容复制到 FLinkKernel->CB中，后续用 CB 和其 W 进行卷积计算
	SdlfFunctionCUDA::BuildFullConnectedArray(FLinkKernel->CB, ImageData, ImageWidth, ImageHeight, ImageDepth, mBatchSize);


	SdlfFunctionCUDA::ConvFullConnected(FLinkKernel->CB, FLinkKernel, OutData, ReluGradient, mBatchSize);
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

void SdlfCalculatorCUDA::CalcFullLink(ConvKernel* FLinkKernel)
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
	SdlfFunctionCUDA::BuildFullConnectedArray(FLinkKernel->CB, mImageOutData, mInWidth, mInHeight, mInChannel, mBatchSize);

	SdlfFunctionCUDA::ConvFullConnected(FLinkKernel->CB, FLinkKernel, OutData, ReluGradient, mBatchSize);
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

extern "C" float* SoftMaxCUDA(SoftMaxKernel * SMK, int mBatchSize, int* mOutChannel, float** mImageOutData);

float* SdlfCalculatorCUDA::SoftMax(SoftMaxKernel* SMK)
{
	// __CUDA__
	SoftMaxCUDA(SMK, mBatchSize, &mOutChannel, &mImageOutData);

	//float* OutData = new float[SMK->Column * mBatchSize];
	//for (int i = 0; i < mBatchSize; i++)
	//{
	//	for (int j = 0; j < SMK->Column; j++)
	//	{
	//		int Pos = i * SMK->Column + j;
	//		OutData[Pos] = SMK->DotProduct(&(mImageOutData[SMK->Row * i]), j);
	//	}
	//	// softmax
	//	int Pos = i * SMK->Column;
	//	float Out[10];
	//	// SdlfFunction::softmax_default(&OutData[Pos], Out, 10);
	//	SdlfFunction::softmax(&OutData[Pos], Out, 10);
	//	memcpy(&OutData[Pos], Out, 10 * sizeof(float));
	//}
	//mOutChannel = SMK->Column;
	//SMK->ApplyLastInput(mImageOutData, mBatchSize * SMK->Row);
	//SAFE_DELETE_ARRAY(mImageOutData);
	//mImageOutData = OutData;

	return mImageOutData;
}

float* SdlfCalculatorCUDA::SoftMax()
{
	float* OutData = new float[mInChannel * mBatchSize];
	for (int i = 0; i < mBatchSize; i++)
	{
		// softmax
		int Pos = i * mInChannel;
		float Out[10];
		//SdlfFunction::softmax_default(&OutData[Pos], Out, 10);
		SdlfFunctionCUDA::softmax(&OutData[Pos], Out, 10);
		memcpy(&OutData[Pos], Out, 10 * sizeof(float));
	}
	mOutChannel = mInChannel;
	mImageOutData = OutData;
	return mImageOutData;
}

void SdlfCalculatorCUDA::Release()
{
	delete this;
}

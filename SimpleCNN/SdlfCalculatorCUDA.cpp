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


// Conv2D��Conv2D_2�������ǣ�Conv2D_2������CPU�ľ����ʽ���ȹ���CB���ټ��㣻
// Conv2D������Ƿ�����һ���Թ���CB�Ĺ��������������ݶ��½�ʱ�������µĴ������ֻ��ǰ�򴫲�
// Conv2D ���ǵ�������ʵ�֡���׼��GPU������̡����Լ��پ�������ھ�����ٿɲο����£�https://blog.csdn.net/qq_40491305/article/details/116236956
void SdlfCalculatorCUDA::Conv2D(float* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel* CK, SdlfActivationFunc ActivationFunc)
{
	mInWidth = mOutWidth = ImageWidth;
	mInHeight = mOutHeight = ImageHeight;
	mInChannel = ImageChannel;

	mOutChannel = CK->ConvKernelCount;
	mBatchSize = BatchSize;
	mPadding = (CK->ConvKernelWidth - 1) >> 1;

	// �������
	// ����Ҫ����CK->ConvKernelCount��һ��ͼ��ʵ�Ƿֿ���CK->ConvKernelCount��ô��������ͼ
	SAFE_DELETE_ARRAY(mImageOutData);
	mImageOutData = new float[BatchSize * ImageWidth * ImageHeight * CK->ConvKernelCount];

	memset(mImageOutData, 0, sizeof(float) * BatchSize * ImageWidth * ImageHeight * CK->ConvKernelCount);

	float* ReluGradientData = new float[mInWidth * mInHeight * BatchSize * CK->ConvKernelCount];

	memset(ReluGradientData, 0, sizeof(float) * mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);

	// ��ʼ������
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

	// ÿ��batch���Էֳ���ô���С��
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
	// ����һ�ε������Ϊ��ε����룬���о������
	mInWidth = mOutWidth;
	mInHeight = mOutHeight;
	mInChannel = CK->ConvKernelChannel;

	// ���channel�Ǿ���˵�channel
	mOutChannel = CK->ConvKernelCount;
	mPadding = (CK->ConvKernelWidth - 1) >> 1;

	// ����Ҫ����CK->ConvKernelCount��һ��ͼ��ʵ�Ƿֿ���CK->ConvKernelCount��ô��������ͼ
	float* OutData = new float[mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount];
	float* ReluGradientData = new float[mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount];

	memset(OutData, 0, sizeof(float) * mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);
	memset(ReluGradientData, 0, sizeof(float) * mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);

	// ��ʼ���������
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

	// ÿ��batch���Էֳ���ô���С��
	Conv2DCUDA_3(d_InData, d_CKC, mImageOutData, mOutWidth, mOutHeight, ReluGradientData, mBatchSize,
		InChannel, OutChannel);

	HANDLE_ERROR(Err);
	Err = cudaFree(d_CKC);
	HANDLE_ERROR(Err);
	Err = cudaFree(d_InData);
	HANDLE_ERROR(Err);
	SAFE_DELETE(CKC);
	//SdlfFunctionCUDA::BuildConvArray(CK->CB, mImageOutData, mInWidth, mInHeight, mInChannel, CK->ConvKernelWidth, CK->ConvKernelHeight, mBatchSize);
	//// ÿ��batch���Էֳ���ô���С��
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

	// �������
	// ����Ҫ����CK->ConvKernelCount��һ��ͼ��ʵ�Ƿֿ���CK->ConvKernelCount��ô��������ͼ
	SAFE_DELETE_ARRAY(mImageOutData);
	mImageOutData = new float[BatchSize * ImageWidth * ImageHeight * CK->ConvKernelCount];

	memset(mImageOutData, 0, sizeof(float) * BatchSize * ImageWidth * ImageHeight * CK->ConvKernelCount);

	float* ReluGradientData = new float[mInWidth * mInHeight * BatchSize * CK->ConvKernelCount];

	memset(ReluGradientData, 0, sizeof(float) * mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);

	// ��ʼ������
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

	// Ϊ CB �����ڴ�
	BuildConvArrayCUDA(d_CBC, d_InData, mInWidth, mInHeight, mInChannel, CK->ConvKernelWidth,
		CK->ConvKernelHeight, mBatchSize);
	// ÿ��batch���Էֳ���ô���С��
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
	// ����һ�ε������Ϊ��ε����룬���о������
	mInWidth = mOutWidth;
	mInHeight = mOutHeight;
	mInChannel = CK->ConvKernelChannel;

	// ���channel�Ǿ���˵�channel
	mOutChannel = CK->ConvKernelCount;
	mPadding = (CK->ConvKernelWidth - 1) >> 1;

	// ����Ҫ����CK->ConvKernelCount��һ��ͼ��ʵ�Ƿֿ���CK->ConvKernelCount��ô��������ͼ
	float* OutData = new float[mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount];
	float* ReluGradientData = new float[mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount];

	memset(OutData, 0, sizeof(float) * mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);
	memset(ReluGradientData, 0, sizeof(float) * mInWidth * mInHeight * mBatchSize * CK->ConvKernelCount);

	// ��ʼ���������
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

	// Ϊ d_CBC �����ڴ�
	BuildConvArrayCUDA(d_CBC, d_InData, mInWidth, mInHeight, mInChannel, CK->ConvKernelWidth,
		CK->ConvKernelHeight, mBatchSize);
	// ÿ��batch���Էֳ���ô���С��

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
		//// ÿ��batch���Էֳ���ô���С��
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
			// �п����ĸ�ֵ��ȣ����ʱ���þ�ֵ�ػ�
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
	// ����4����Ϊ�ػ�����������ǿ��һ�߳���2
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

	// ȫ����֮�������ͼƬ��ȸ��߶�֮��ģ�ȫ��û���ˣ�Ӧ����ֻ��һ��channel�ˡ�
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
	// �� imgData ���ݸ��Ƶ� FLinkKernel->CB�У������� CB ���� W ���о������
	SdlfFunctionCUDA::BuildFullConnectedArray(FLinkKernel->CB, ImageData, ImageWidth, ImageHeight, ImageDepth, mBatchSize);


	SdlfFunctionCUDA::ConvFullConnected(FLinkKernel->CB, FLinkKernel, OutData, ReluGradient, mBatchSize);
	FLinkKernel->ApplyImageInputWidthAndHeight(mOutWidth, mOutHeight);

	// �������Ǳ����1����0�ˣ�һ��ͼƬ���Ѿ�ȫ�������N����
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

	// ȫ����֮�������ͼƬ��ȸ��߶�֮��ģ�ȫ��û���ˣ�Ӧ����ֻ��һ��channel�ˡ�
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

	// ���⣬������ֱ��DropOut���ˡ���Ȼ��Ҫ�������ڰ���
	// SdlfFunction::DropOut(OutData, DropOutGradient, mBatchCount * FLinkKernel->ConvKernelCount, DROP_OUT_PARAM);
	// �������Ǳ����1����0�ˣ�һ��ͼƬ���Ѿ�ȫ�������N����
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

/***********************************************
 * File: KernelCUDA.h
 *
 * Author: CHB
 * Date: 2021-04-20
 *
 * Purpose:
 *
 *
 **********************************************/

#pragma once

#include "Common.h"
#include <iostream>
#include <string.h>
#include "SdlfFunctionCUDA.h"
#include "KernelCPU.h"

#define				DROP_OUT_PARAM				0.3f
#define				STEP						0.1f

#define APPLY_DATA(V)														\
void Apply##V(float* Data, int Len)											\
{																			\
	if(V == nullptr)														\
	{																		\
		V = new float[Len];													\
	}																		\
	memcpy(V, Data, Len * sizeof(float));									\
}


struct ConvBlocksCUDA
{
	float* ConvArray;
	int ArrayLen;
	int cBatchSize;
	int cImgSize;
	ConvBlocks* CB;
	ConvBlocksCUDA(ConvBlocks* CBs, int BatchSize, int ImgSize)
	{
		cudaError_t Err; 
		Err = CUDA_MALLOC(ConvArray, CBs->ArrayLen * BatchSize * ImgSize *  sizeof(float));
		HANDLE_ERROR(Err);
		Err = cudaMemcpy(ConvArray, CBs->ConvArray, CBs->ArrayLen * BatchSize * ImgSize * sizeof(float), cudaMemcpyHostToDevice);
		HANDLE_ERROR(Err);
		ArrayLen = CBs->ArrayLen;
		CB = CBs;
		cBatchSize = BatchSize;
		cImgSize = ImgSize;
	}
	~ConvBlocksCUDA()
	{
		cudaError_t Err;
		Err = cudaMemcpy(CB->ConvArray, ConvArray, cBatchSize * ArrayLen * cImgSize * sizeof(float), cudaMemcpyDeviceToHost);
		HANDLE_ERROR(Err);
		Err = cudaFree(ConvArray);
		HANDLE_ERROR(Err);
		CB = nullptr;
		ArrayLen = 0;
		cBatchSize = 0;
	}
};

struct ConvKernelCUDA
{
	int ConvKernelWidth;
	int ConvKernelHeight;
	int ConvKernelChannel;
	int ConvKernelCount;  // kernel ������Ϊ ��� OutChannels ��ֵ

	float* W;
	float* B;

	// ���򴫲��л��õ�
	// ...
	int ImageInputWidth;
	int ImageInputHeight;
	ConvBlocks* CB;
	float* WRotate180;				// �����󵼵�ʱ���õ�����ת180��
	float* DropOutGradient;
	float* ImageMaxPoolGradientData;
	float* ImageReluGradientData;
	ConvKernel* cCK;

	ConvKernelCUDA(ConvKernel* CK)
	{
		cudaError_t Err;
		ConvKernelWidth = CK->ConvKernelWidth;
		ConvKernelHeight = CK->ConvKernelHeight;
		ConvKernelChannel = CK->ConvKernelChannel;
		ConvKernelCount = CK->ConvKernelCount;
		// W = CK->W;
		// WRotate180 = CK->WRotate180;
		// B = CK->B;
		CB = CK->CB;
		ImageInputWidth = CK->ImageInputWidth;
		ImageInputHeight = CK->ImageInputHeight;
		DropOutGradient = CK->DropOutGradient;
		ImageMaxPoolGradientData = CK->ImageMaxPoolGradientData;
		ImageReluGradientData = CK->ImageReluGradientData;
		cCK = CK;
		int W_Size = ConvKernelWidth * ConvKernelHeight * ConvKernelChannel * ConvKernelCount * sizeof(float);
		int B_Size = ConvKernelCount * sizeof(float);
		Err = CUDA_MALLOC(W, W_Size);
		HANDLE_ERROR(Err);
		Err = cudaMemset(W, 0, W_Size);
		HANDLE_ERROR(Err);
		Err = CUDA_MALLOC(B, B_Size);
		HANDLE_ERROR(Err);
		Err = CUDA_MALLOC(WRotate180, W_Size);
		HANDLE_ERROR(Err);
		Err = cudaMemcpy(W, CK->W, W_Size, cudaMemcpyHostToDevice);
		HANDLE_ERROR(Err);
		Err = cudaMemcpy(B, CK->B, B_Size, cudaMemcpyHostToDevice);
		HANDLE_ERROR(Err);
		Err = cudaMemcpy(WRotate180, CK->WRotate180, W_Size, cudaMemcpyHostToDevice);
		HANDLE_ERROR(Err);
	}
	~ConvKernelCUDA()
	{
		cudaError_t Err;
		ConvKernel* CK = cCK;
		CK->ConvKernelWidth = ConvKernelWidth;
		CK->ConvKernelHeight = ConvKernelHeight;
		CK->ConvKernelChannel = ConvKernelChannel;
		CK->ConvKernelCount = ConvKernelCount;
		int W_Size = ConvKernelWidth * ConvKernelHeight * ConvKernelChannel * ConvKernelCount * sizeof(float);
		int B_Size = ConvKernelCount * sizeof(float);
		// CK->W = W;
		// CK->WRotate180 = WRotate180;
		// CK->B = B;
		Err = cudaMemcpy(CK->W, W, W_Size, cudaMemcpyDeviceToHost);
		HANDLE_ERROR(Err);
		Err = cudaFree(W);
		HANDLE_ERROR(Err);
		Err = cudaMemcpy(CK->B, B, B_Size, cudaMemcpyDeviceToHost);
		HANDLE_ERROR(Err);
		Err = cudaFree(B);
		HANDLE_ERROR(Err);
		Err = cudaMemcpy(CK->WRotate180, WRotate180, W_Size, cudaMemcpyDeviceToHost);
		HANDLE_ERROR(Err);
		Err = cudaFree(WRotate180);
		HANDLE_ERROR(Err);
		CK->CB = CB;
		CK->ImageInputWidth = ImageInputWidth;
		CK->ImageInputHeight = ImageInputHeight;
		CK->DropOutGradient = DropOutGradient;
		CK->ImageMaxPoolGradientData = ImageMaxPoolGradientData;
		CK->ImageReluGradientData = ImageReluGradientData;
		cCK = nullptr;
	}

	APPLY_DATA(DropOutGradient);
	APPLY_DATA(ImageMaxPoolGradientData);
	APPLY_DATA(ImageReluGradientData);

	void InitializeDataToRandom(float fMin, float fMax)
	{
		int Len = ConvKernelWidth * ConvKernelHeight * ConvKernelChannel * ConvKernelCount;
		for (int i = 0; i < Len; i++) {
			W[i] = SdlfFunctionCUDA::RangeRandom(fMin, fMax);
		}
		for (int i = 0; i < ConvKernelCount; i++) {
			B[i] = SdlfFunctionCUDA::RangeRandom(fMin, fMax);
		}
	}

	void InitializeData(float w, float b)
	{
		int Len = ConvKernelWidth * ConvKernelHeight * ConvKernelChannel * ConvKernelCount;
		for (int i = 0; i < Len; i++)
		{
			W[i] = w;
		}
		for (int i = 0; i < ConvKernelCount; i++)
		{
			B[i] = b;
		}
	}

	// ���������ȫ���ӣ�Ҳ�����Ǿ�������ռ��㷽ʽ���޲�ͬ
	// ���ȫ���ӣ���ʵҲ���Ǹ����������һ��ͼƬ��N��֮�����Ϊ7 * 7 * 64��Ҫ��һ��ȫ���ӣ���ô������һ��7 * 7 * 64 * 1024������1024������ˣ�ÿ������˸�֮ǰ�������ˣ��õ�һ����
	// ������һ��ͼƬ���ͱ����1024������
	float Conv2D(float* Data, int CountIndex)
	{
		float Sum = 0;
		int Pos = ConvKernelWidth * ConvKernelHeight * ConvKernelChannel * CountIndex;
		for (int i = 0; i < ConvKernelWidth * ConvKernelHeight * ConvKernelChannel; i++)
		{
			Sum += W[Pos + i] * Data[i];
		}
		Sum += B[CountIndex];

		return Sum;
	}

	void ApplyImageInputWidthAndHeight(int Width, int Height)
	{
		ImageInputWidth = Width;
		ImageInputHeight = Height;
	}
	// ��һ�����ٶȿ�㣬������һ���򵥵ĳ˷�
	void CalcFullLinkDropOutAndReluGradient(float* GradientData, int BatchSize)
	{
		int Len = BatchSize * ConvKernelCount;
		for (int i = 0; i < Len; i++)
		{
			GradientData[i] = GradientData[i] * ImageReluGradientData[i];  // * DropOutGradient[i];
		}
		return;
	}

	// ȫ������
	// ��Ҫ��1 * 1024��ÿ����������һ������ˣ��õ�Width * Height * Depth��ͼ��һ���󵼣���1024��ͼ��ӡ����⣬��Ҫ���ǵ�batch count
	// ConvKernelCount == 1024��GradientData�ĳ�����1024 * BatchCount
	// ��ǰ�󵼣���Ҫ���ǵ�Ȩ�ء������Ȩ�أ���1024��w��ͣ�����Ҫ����������
	// OutData��С��Width * Height * Depth * BatchCount.
	// ȫ���ӣ����û���������˴�С��ͼƬ�Ĵ�С��һ����
	void CalcFullLinkGradient(float* GradientData, int BatchSize, float* OutData)
	{
		// �����7 * 7 * 64
		int ImageSize = ConvKernelWidth * ConvKernelHeight * ConvKernelChannel;
		memset(OutData, 0, sizeof(float) * ImageSize * BatchSize);
		for (int i = 0; i < BatchSize; i++)
		{
			for (int j = 0; j < ConvKernelCount; j++)
			{
				int Pos = i * ConvKernelCount + j;
				for (int k = 0; k < ImageSize; k++)
				{
					OutData[ImageSize * i + k] += (GradientData[Pos] * W[j * ImageSize + k]);
				}
			}
		}
	}

	// ����ȫ���Ӳ���
	// GradientData�ĳ�����1024 * BatchCount
	// ImageInputData��ȫ�����ϴ������������ݣ�ʵ�����ݳ�����ConvKernelWidth * ConvKernelHeight * ConvKernelChannel * BatchCount
	// ȫ���ӵ�ʵ�ʣ���һ��w * h * d��ͼƬ�����һ��w * h * d�ľ���ˣ����һ������ȫ����������1024������ˣ����Ա����1024������
	// �������µ�ʵ�ʣ���w * x + b���w��ƫ������ʵ����x�����Ե�����ʵ��GradientData * x��
	// �����x����LastInput��Ҳ�����ϴγػ�֮��������
	// ���²�������Ҫ����batch size
	void UpdateFullLinkParameter(float* GradientData, int BatchSize, float step)
	{
		int ImageSize = ConvKernelWidth * ConvKernelHeight * ConvKernelChannel;
		for (int i = 0; i < BatchSize; i++)
		{
			for (int j = 0; j < ConvKernelCount; j++)
			{
				// ����B
				B[j] -= step * GradientData[i * ConvKernelCount + j] / float(BatchSize);
				// ����W
				for (int k = 0; k < ImageSize; k++)
				{
					W[j * ImageSize + k] -= step * GradientData[i * ConvKernelCount + j] * CB->ConvArray[i * CB->ArrayLen + k] / float(BatchSize);
				}
			}

		}
	}

	void CalcConv2DMaxPoolAndReluGradient(float* GradientData, float* OutputData)
	{
		for (int i = 0; i < ImageInputHeight / 2; i++)
		{
			for (int j = 0; j < ImageInputWidth / 2; j++)
			{
				int Pos1 = i * 2 * ImageInputWidth + j * 2;
				int Pos2 = i * 2 * ImageInputWidth + j * 2 + 1;
				int Pos3 = (i * 2 + 1) * ImageInputWidth + j * 2;
				int Pos4 = (i * 2 + 1) * ImageInputWidth + j * 2 + 1;

				int Pos = i * ImageInputWidth / 2 + j;
				OutputData[Pos1] = GradientData[Pos] * ImageMaxPoolGradientData[Pos1] * ImageReluGradientData[Pos1];
				OutputData[Pos2] = GradientData[Pos] * ImageMaxPoolGradientData[Pos2] * ImageReluGradientData[Pos2];
				OutputData[Pos3] = GradientData[Pos] * ImageMaxPoolGradientData[Pos3] * ImageReluGradientData[Pos3];
				OutputData[Pos4] = GradientData[Pos] * ImageMaxPoolGradientData[Pos4] * ImageReluGradientData[Pos4];
			}
		}
	}

	// ����MaxPool��Relu��ǰ�󵼣�����һ��д����Ϊ��������
	// GradientData�Ĵ�С��ImageInputWidth * ImageInputHeight * ConvKernelCount * BatchCount / 4
	// ����4����Ϊ����ͼƬMaxPool֮�󣬿����ʵһ�߳���2
	// MaxPool������֮�󣬵õ������ݻ����ı������������С��ImageInputWidth * ImageInputHeight * ConvKernelCount * BatchCount
	void CalcConv2DMaxPoolAndReluGradient(float* GradientData, float* OutputData, int BatchSize)
	{
		int ImageSize = ImageInputHeight * ImageInputWidth;
		for (int i = 0; i < BatchSize * ConvKernelCount; i++)
		{
			CalcConv2DMaxPoolAndReluGradient(&GradientData[ImageSize * i / 4], &OutputData[ImageSize * i]);
		}
	}

	// ��ֵ����˵�������Ҫ��ԭ�������ת180�ȡ�����˴�С�Ȳ�����Ȼ��ȡ�
	void ApplyConvKernelAndRotate180()
	{
		int HalfConv = ConvKernelWidth / 2;
		int ImageSize = ConvKernelHeight * ConvKernelWidth;
		for (int i = 0; i < ConvKernelCount; i++)
		{
			for (int j = 0; j < ConvKernelChannel; j++)
			{
				for (int m = 0; m < ConvKernelHeight; m++)
				{
					for (int n = 0; n < ConvKernelWidth; n++)
					{
						// ������ת�������ĵ���ת��������Ҫ�����ĵ�Ϊ����ԭ�㣬��������ϵ��
						// �þ�������һ����ת����ʵ�ܼ򵥣�����ԭ��������(x,y)����ת180�ȵ���������(-x, -y),cos(180) = -1
						int x = n - HalfConv;
						int y = m - HalfConv;
						int _x = -x + HalfConv;
						int _y = -y + HalfConv;
						int Rotate180Pos = ImageSize * ConvKernelChannel * i + ImageSize * j + ConvKernelWidth * _y + _x;
						int OriginalPos = ImageSize * ConvKernelChannel * i + ImageSize * j + ConvKernelWidth * m + n;
						WRotate180[OriginalPos] = W[Rotate180Pos];
					}
				}
			}
		}
	}


	// ���²��֣���ʵ������ע�͵��ĵ��㷨����д��һ�顣Ϊʲô��ô�ɣ���Ҫ�Ǿ�����������д�������׶�һЩ���ҿ�ʼ�õ�������������������Ч��̫���ˣ��Ļ�����ָ�룬��ʵЧ�ʻ�����������㷨��˵���˶����ᡣ
	// OutData�Ĵ�С����ImageInputWidth * ImageInputHeight * ConvKernelChannel
	void CalcConv2DGradientPerImage(float* GradientData, float* OutData, int WIndex)
	{
		// �������ͼƬ��������Ҫ���һ������ˣ�����ͼƬ��14 * 14���������5 * 5 * 32����ô��������Ҫ����ȥ���32��5 * 5�ľ���ˣ��õ�32��14 * 14ͼƬ�����ӳ�һ��14 * 14 * 32��ͼƬ
		// �����Ҫ��64��14 * 14 * 32��ͼƬ����ͳ�һ��
		int ConvLayerSize = ConvKernelWidth * ConvKernelHeight;
		int ConvKernelSize = ConvLayerSize * ConvKernelChannel;
		float* WRotate_P = WRotate180 + ConvKernelSize * WIndex;
		ConvBlocks* CB = new ConvBlocks;
		SdlfFunctionCUDA::BuildConvArray(CB, GradientData, ImageInputWidth, ImageInputHeight, 1, ConvKernelWidth, ConvKernelHeight, 1);
		// ����˸���
		int CKCount = ImageInputWidth * ImageInputHeight;
		for (int i = 0; i < ConvKernelChannel; i++)
		{
			int StartIndex = i * ConvKernelWidth * ConvKernelHeight;
			float* WRotate_P_I = WRotate_P + i * ConvLayerSize;

			for (int j = 0; j < CKCount; j++)
			{
				float Sum = SdlfFunctionCUDA::DotProduct(WRotate_P_I, &CB->ConvArray[j * CB->ArrayLen], ConvLayerSize);
				OutData[i * CKCount + j] += Sum;
			}
		}

		SAFE_DELETE(CB);
	}

	void CalcConv2DGradientPerBatch(float* GradientData, int BatchIndex, float* OutData)
	{
		// ��ǰ�����ʱ��һ��Batch�����ConvKernelCount�Ǹ����ͼƬ�����ﷴ��������Ҫ��������ͼƬ����������һ��batch�����ݣ�Ȼ�����
		int Image2DSize = ImageInputWidth * ImageInputHeight;
		for (int i = 0; i < ConvKernelCount; i++)
		{
			CalcConv2DGradientPerImage(GradientData + Image2DSize * i, OutData, i);
		}
	}
	// �����ǰ��
	// ��ǰ�󵼣���Ҫ�ȰѾ������ת180�ȡ������ĵ����������ϸ���ܡ�
	// ��������Ѿ���ת��ɣ����ұ��浽�󵼾����
	// �����ʱ�����۾���˵�channel�Ƿ�Ϊ1����Ȼ���һ��depthΪ1��ͼƬ��������ֻ������ѵ�������������֪����
	// ��ˣ�����󵼵�ʱ����Ҫ����һ��ͼƬ������������ת��ľ���ˣ��õ�N��N�Ǿ���˵�channel����ͼƬ��
	// ����˵��������������14 * 14 * 32��ͼƬ���������5 * 5 * 32 * 64�����֮�󣬻�����14 * 14 * 64��ͼƬ��
	// ���Է����󵼣���64��14 * 14��ͼƬ��ȥ���64��5 * 5 * 32�ľ���ˣ�����ľ���ˣ�����ת���ģ����õ�64��14 * 14 * 32��ͼƬ�����64��ͼƬ����ͣ��õ�һ�š�
	void CalcConv2DGradient(float* GradientData, int BatchCount, float* OutData)
	{
		int Image2DSize = ImageInputWidth * ImageInputHeight;
		int ImageSize = Image2DSize * ConvKernelChannel;
		int ImageOutBatchSize = Image2DSize * ConvKernelCount;
		int ImageInBatchSize = ImageSize;

		for (int i = 0; i < BatchCount; i++)
		{
			CalcConv2DGradientPerBatch(GradientData + ImageOutBatchSize * i, i, OutData + ImageInBatchSize * i);
		}
	}

	// �����ʱ��ÿһ��ͼƬ������һ������ͼƬ������14 * 14 * 32�������һ������ˣ�����5 * 5 * 32 * 64���õ��ġ�
	// һ������ͼƬ�����ﰴ����õ�64�����ͼƬ������ÿһ��ͼƬ����������Ҫȥ���¶�Ӧλ�õľ���ˣ������5��ͼƬ��ȥ���µ��������˵Ĳ���
	// ���ھ�������ǳ��࣬����һ��ͼƬ������Width * Height��ô��Σ�����һ������������B����Ҫ��������ô��Σ�
	// ���⣬����Ҫ����BatchSize
	void UpdateConv2DParameterPerImage(int BatchSize, int BatchIndex, int ConvIndex, float* GradientData, float step)
	{
		int BlockCount = ImageInputWidth * ImageInputHeight;
		int StartIndex = BlockCount * BatchIndex;
		int ConvBlockSize = ConvKernelWidth * ConvKernelHeight * ConvKernelChannel;
		int ImageSize = BlockCount;
		for (int i = 0; i < ImageSize; i++)
		{
			B[ConvIndex] -= (step * GradientData[i] / float(BatchSize));
			for (int j = 0; j < ConvBlockSize; j++)
			{
				// ÿ��������Ҫ�ֱ����һ���������ô��С��һ��С��
				W[ConvIndex * ConvBlockSize + j] -= (step * GradientData[i] * CB->ConvArray[(StartIndex + i) * CB->ArrayLen + j] / float(BatchSize));
			}
		}
	}
	// ���¾������
	// GradientData��С��ImageInputWidth * ImageInputHeight * ConvKernelCount * BatchCount
	// ��������Կ�����ConvKernelCount * BatchCount��ô����ͼƬ��ÿ��ͼƬ��ÿ�����أ���Ҫ����һ�β�����
	// ÿ��ͼƬ�Ĵ�С����ʵ��ImageInputWidth * ImageInputHeight * ConvKernelChannel
	// ���µķ�ʽ����ʵ������ƫ������˵ķ�ʽ����������ͼƬ��Ӧλ�õ����ء�
	// ���磬�������14 * 14 * 32�ģ���64����ÿ��ͼƬÿ�����أ���Ҫ����14 * 14 * 32��������
	// ���ڣ�������������طǳ��࣬ÿ�����أ�����Ҫ������ͼƬ��һ�������ˣ�
	// ��ˣ����ŵķ�������ǰ�Ȱ�����ͼƬ���ָ��һ��һ������˴�С��С�飬����ÿһ�����ͼƬ���ص�ʱ�����������е�С����ˣ���ȥ���¾���˲���
	// ����ͼƬ��Ե�����ز�0��
	void UpdateConv2DParameter(float* GradientData, int BatchCount, float step)
	{
		// ��һ�����Ȱ�����ͼƬ���ֳ�һ��һ��С�顣ÿ��С��Ĵ�С����ConvKernelWidth * ConvKernelHeight * ConvKernelChannel
		// һ��ͼƬ�ָܷ�Ĵ�С����ImageInputWidth * ImageInputHeight��ô�����������Ҫ����channel����Ϊ����Ҳ����������˵ġ�
		// ���ԣ�����ֳɵ�С��������ImageInputWidth * ImageInputHeight * BatchCount��
		// ������һ�£�ʹ�����ֿ��ٷ�ʽ��һ���󵼵��ڴ�ʹ�ã������64batch size����Ҫ40M������һ���൱�ֲ��������ˣ���Ϊ�����Ǻ�С��ͼ��ֻ��14 * 14.��ͼ�Ļ������ַ�ʽ��������ѧ��

		int Image2DSize = ImageInputWidth * ImageInputHeight;
		int ImageBlockSize = Image2DSize * ConvKernelCount;

		// һ��һ��ͼƬ������ȥ����W�������ܹ���ImageCount��ô����ͼƬ
		for (int i = 0; i < BatchCount; i++)
		{
			for (int j = 0; j < ConvKernelCount; j++)
			{
				UpdateConv2DParameterPerImage(BatchCount, i, j, &GradientData[i * ImageBlockSize + j * Image2DSize], step);
			}
		}
	}
};


// ȫ����֮��softmax֮ǰ����Ҫ��1 * 1024������ת����1 * 10�����ݣ���softmax
// �������������ø��򵥵ľ���˷�������һ��1024 * 10�ľ���
// ��������Row��1024��Column��10
// ���ﲻ̫֪��Ӧ����������������������
struct SoftMaxKernelCUDA
{
	int Row;  // InChannel, 1024
	int Column;	// OutChannel, 10

	float* W;
	float* B;
	// ������Ȩ�ؼ����õ�
	float* WSum;
	// д������ȫ�Ǹ��������롣����ȷʵ����ǰ����Ƶ�ʱ����Ϥ���ִ�֮��д�ġ�ʵ�����벻�����õķ����ˡ�����ǰ�����ϡ�ã�һ��дһ���ֲ��ĸо�����
	// ����Ĵ�СӦ����1024 * BatchCount
	float* LastInput;
	SoftMaxKernel* cSMK;
	SoftMaxKernelCUDA(SoftMaxKernel* SMK)
	{
		cSMK = SMK;
		Row = SMK->Row;
		Column = SMK->Column;
		cudaError_t Err;
		Err = CUDA_MALLOC(W, Row * Column * sizeof(float));
		HANDLE_ERROR(Err);
		Err = CUDA_MALLOC(B, Column * sizeof(float));
		HANDLE_ERROR(Err);
		Err = CUDA_MALLOC(WSum, Row * sizeof(float));
		HANDLE_ERROR(Err);
		Err = cudaMemcpy(W, SMK->W, Row * Column * sizeof(float), cudaMemcpyHostToDevice);
		HANDLE_ERROR(Err);
		Err = cudaMemcpy(B, SMK->B, Column * sizeof(float), cudaMemcpyHostToDevice);
		HANDLE_ERROR(Err);
		Err = cudaMemcpy(WSum, SMK->WSum, Row * sizeof(float), cudaMemcpyHostToDevice);
		HANDLE_ERROR(Err);
		LastInput = nullptr;
	}
	~SoftMaxKernelCUDA()
	{
		SoftMaxKernel* SMK = cSMK;
		cudaError_t Err;
		Err = cudaMemcpy(SMK->W, W, Row * Column * sizeof(float), cudaMemcpyDeviceToHost);
		HANDLE_ERROR(Err);
		Err = cudaFree(W);
		HANDLE_ERROR(Err);
		Err = cudaMemcpy(SMK->B, B, Column * sizeof(float), cudaMemcpyDeviceToHost);
		HANDLE_ERROR(Err);
		Err = cudaFree(B);
		HANDLE_ERROR(Err);
		Err = cudaMemcpy(SMK->WSum, WSum, Row * sizeof(float), cudaMemcpyDeviceToHost);
		HANDLE_ERROR(Err);
		Err = cudaFree(WSum);
		HANDLE_ERROR(Err);
		cSMK = nullptr;
		/*SAFE_DELETE_ARRAY(W);
		SAFE_DELETE_ARRAY(B);
		SAFE_DELETE_ARRAY(WSum);*/
		SAFE_DELETE_ARRAY(LastInput);
	}
	APPLY_DATA(LastInput)

void InitializeData(float w, float b)
	{
		int Len = Row * Column;
		for (int i = 0; i < Len; i++)
		{
			W[i] = w;
		}
		for (int i = 0; i < Column; i++)
		{
			B[i] = b;
		}
	}

	void InitializeDataToRandom(float fMin, float fMax)
	{
		int Len = Row * Column;
		for (int i = 0; i < Len; i++)
		{
			W[i] = SdlfFunctionCUDA::RangeRandom(fMin, fMax);
		}
		for (int i = 0; i < Column; i++)
		{
			B[i] = SdlfFunctionCUDA::RangeRandom(fMin, fMax);
		}
	}

	float DotProduct(float* Data, int ColIndex)
	{
		float Sum = 0.0f;

		for (int i = 0; i < Row; i++)
		{
			Sum += Data[i] * W[ColIndex * Row + i];
		}
		Sum += B[ColIndex];
		return Sum;
	}
	// ���²��������ݷ�����ʽ�󵼸���
	// ʵ��ѵ����ʱ�򣬼��������sigma(OutputData[i] * W[i]) + B
	// ���ԣ��󵼸��²�����ʱ����ʵÿ������W -= step * (GradientData[i] * OutputData[i]);
	// ���ԣ�����OutputData��֮ǰ��ûsoftmax֮ǰ��1 * 1024�����ݡ�GradientData��loss��֮���1 * 10������
	// ���²�������Ҫ��������
	void UpdateParameter(float* GradientData, float* OutputData, float step, int BatchSize)
	{
		// �ȸ���B
		for (int i = 0; i < Column; i++)
		{
			B[i] -= step * GradientData[i] / float(BatchSize);
			// ����W
			int Pos = i * Row;
			for (int j = 0; j < Row; j++)
			{
				W[Pos + j] -= (OutputData[j] * GradientData[i] * step / float(BatchSize));
			}
		}
	}
	void UpdateParameter(float* GradientData, int BatchSize, float step)
	{
		for (int i = 0; i < BatchSize; i++)
		{
			UpdateParameter(&GradientData[i * Column], &LastInput[i * Row], step, BatchSize);
		}
	}
	// ��ʽ�󵼣���ǰ�ƽ�
	// ��ʽ�󵼵�ʱ��Ӧ����1 * 10�����ݣ�ÿһ������������1024 * 1�����ݣ��õ�һ��1024 * 1�����顣Ȼ��ʮ��1024 * 1��������ӣ��õ����յ�1 * 1024(1024 * 1)������
	// o = w1 * x1 + w2 * x2 + w3 * x3��������w�󵼣��Ǿ���x����x�󵼣��Ǿ���w�����ԣ����ﷴ���󵼣����ǳ���w
	// GradientData��1 * 10�ķ��������ݣ�OutputData����Ҫ����õ���1 * 1024������
	// ���Ҫ���ǵ�Ȩ�ط��䡣Ҳ����˵����͵�ʱ����Ҫ����w1 + w2 + w3 + ���� + w10
	void CalcGradient(float* GradientData, float* OutputData)
	{
		memset(OutputData, 0, Row * sizeof(float));
		for (int i = 0; i < Column; i++)
		{
			for (int j = 0; j < Row; j++)
			{
				OutputData[j] += GradientData[i] * W[i * Row + j];
			}
		}
		/*
		for (int j = 0; j < Row; j++)
		{
			OutputData[j] /= WSum[j];
		}
		*/
	}

	void CalcGradient(int BatchSize, float* GradientData, float* OutputData)
	{
		// ��ʼ����֮ǰ���ȰѲ�����ͣ�����Ȩ�ؼ���Ҫ�õ�
		memset(WSum, 0, sizeof(float) * Row);
		for (int i = 0; i < Row; i++)
		{
			for (int j = 0; j < Column; j++)
			{
				WSum[i] += W[j * Row + i];
			}
		}
		for (int i = 0; i < BatchSize; i++)
		{
			CalcGradient(&GradientData[i * Column], &OutputData[i * Row]);
		}
	}
};
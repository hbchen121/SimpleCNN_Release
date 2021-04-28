/***********************************************
 * File: SdlfFunctionCUDA.cpp
 *
 * Author: CHB
 * Date: 2020-04-18
 *
 * Purpose:定义常用函数
 *
 *
 **********************************************/
#include <math.h>
#include <algorithm>
#include "SdlfFunctionCUDA.h"
#include "KernelCUDA.h"
#include "Common.h"

float SdlfFunctionCUDA::MaxInPool(float* Data, int Num, int& MaxIndex)
{
    // 简单冒泡一轮
    float MaxValue = Data[0];
    MaxIndex = 0;
    for (int i = 1; i < Num; i++)
    {
        if (MaxValue < Data[i])
        {
            MaxValue = Data[i];
            MaxIndex = i;
        }
    }

    return MaxValue;
}

float SdlfFunctionCUDA::DotProduct(float* Data1, float* Data2, int Num)
{
    float Sum = 0.0f;
    for (int i = 0; i < Num; i++)
    {
        Sum += Data1[i] * Data2[i];
    }

    return Sum;
}

float SdlfFunctionCUDA::UnitRandom()
{
    static bool FirstRun = true;
    if (FirstRun) {
        srand((unsigned)time(NULL));
        FirstRun = false;
    }
    return float(rand()) / float(RAND_MAX);
}

float SdlfFunctionCUDA::RangeRandom(float Min, float Max)
{
    return Min + (Max - Min) * UnitRandom();
}

double SdlfFunctionCUDA::Pow(double x, double y)
{
    return pow(x, y);
}

double SdlfFunctionCUDA::Exp(double n)
{
    return exp(n);
}

double SdlfFunctionCUDA::ln(double n)
{
    return log(n);
}

double SdlfFunctionCUDA::lg(double n)
{
    return log10(n);
}

double SdlfFunctionCUDA::log_m_n(double m, double n)
{
    return ln(n) / ln(m);
}

void SdlfFunctionCUDA::softmax(float* InArrayDest, float* OutArray, int Num)
{
    float sum = 0;
    for (int i = 0; i < Num; i++) {
        OutArray[i] = (float)Exp(InArrayDest[i]);
        sum += OutArray[i];
    }
    for (int i = 0; i < Num; i++)
    {
        OutArray[i] /= sum;
    }

    return;
}


void SdlfFunctionCUDA::BuildConvArrayPerBatch(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int ConvWidth, int ConvHeight, int BatchSizeIndex)
{
    // 每个小块的大小，是卷积核的大小
    int ConvLen = ConvWidth * ConvHeight * ImageDepth;
    // 遍历图片，一块一块的生成小块
    // 每个Batch有这么多个小块
    int ImgSizePerChannel = ImageHeight * ImageWidth;
    int HalfConv = ConvWidth >> 1;
    float* ConvQuad = new float[ConvLen];
    for (int i = 0; i < ImageHeight; i++) {
        for (int j = 0; j < ImageWidth; j++) {
            // 开始填充quad
            for (int k = 0; k < ImageDepth; k++) {
                for (int m = 0; m < ConvHeight; m++) {
                    for (int n = 0; n < ConvWidth; n++) {
                        int ConvPos = k * ConvHeight * ConvWidth + m * ConvWidth + n;
                        int x_ = n - HalfConv;
                        int y_ = m - HalfConv;
                        int x = j + x_;
                        int y = i + y_;
                        if (x < 0 || x >= ImageWidth || y < 0 || y >= ImageHeight) {
                            ConvQuad[ConvPos] = 0;
                        }
                        else {
                            int Pos = ImgSizePerChannel * k + y * ImageWidth + x;
                            ConvQuad[ConvPos] = ImageData[Pos];
                        }
                    }
                }
            }
            memcpy(CB->ConvArray + (ImgSizePerChannel * BatchSizeIndex + i * ImageWidth + j) * ConvLen, ConvQuad, sizeof(float) * ConvLen);
        }
    }
    SAFE_DELETE_ARRAY(ConvQuad);
}

void SdlfFunctionCUDA::BuildConvArray(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int ConvWidth, int ConvHeight, int BatchSize)
{
    int ConvLen = ConvWidth * ConvHeight * ImageDepth;
    int ImageSize = ImageWidth * ImageHeight * ImageDepth;
    if (CB->ConvArray == nullptr) {
        CB->ConvArray = new float[ImageWidth * ImageHeight * BatchSize * ConvLen];
        CB->ArrayLen = ConvLen;
    }
    // 按 batch 划分 convArray 进行计算
    for (int i = 0; i < BatchSize; i++) {
        int Pos = i * ImageSize;
        BuildConvArrayPerBatch(CB, &ImageData[Pos], ImageWidth, ImageHeight, ImageDepth, ConvWidth, ConvHeight, i);
    }
    return;
}

void SdlfFunctionCUDA::BuildFullConnectedArrayPerBatch(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int BatchIndex)
{
    int ImgSize = ImageWidth * ImageHeight * ImageDepth;
    memcpy(CB->ConvArray + BatchIndex * ImgSize, ImageData, sizeof(float) * ImgSize);
}

void SdlfFunctionCUDA::BuildFullConnectedArray(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int BatchSize)
{
    int ImgSize = ImageWidth * ImageHeight * ImageDepth;
    if (CB->ConvArray == nullptr) {
        CB->ConvArray = new float[ImgSize * BatchSize];
        CB->ArrayLen = ImgSize;
    }
    for (int i = 0; i < BatchSize; i++) {
        BuildFullConnectedArrayPerBatch(CB, ImageData + i * ImgSize, ImageWidth, ImageHeight, ImageDepth, i);
    }
}

extern "C" float* ConvFullConnectedCUDA(ConvBlocks * CB, ConvKernel * CK, float* ImageOutData, float* ReluOutData, int BatchSize);

void SdlfFunctionCUDA::ConvFullConnected(ConvBlocks* CB, ConvKernel* CK, float* ImageOutData, float* ReluOutData, int BatchSize)
{
    // __CUDA__
    ConvFullConnectedCUDA(CB, CK, ImageOutData, ReluOutData, BatchSize);
    // 对于全连接来说，一张图片生成一个卷积，所以这里的卷积数，就是batch数
    /*for (int i = 0; i < BatchSize; i++)
    {
        for (int j = 0; j < CK->ConvKernelCount; j++)
        {
            float ConvResult = CK->Conv2D(&CB->ConvArray[i * CB->ArrayLen], j);
            if (ConvResult > 0.0f)
            {
                ImageOutData[i * CK->ConvKernelCount + j] = ConvResult;
                ReluOutData[i * CK->ConvKernelCount + j] = 1.0f;
            }
        }
    }*/
}

extern "C" float* Conv2DCUDA(ConvBlocks * CB, int ConvBlocksPerBatch, ConvKernel * CK, float* ImageOutData, int ImageSize2D, float* ReluOutData, int BatchSize);

void SdlfFunctionCUDA::Conv2D(ConvBlocks* CB, int ConvBlocksPerBatch, ConvKernel* CK, float* ImageOutData, int ImageSize2D, float* ReluOutData, int BatchSize)
{
    // __CUDA__
    Conv2DCUDA(CB, ConvBlocksPerBatch, CK, ImageOutData, ImageSize2D, ReluOutData, BatchSize);
    // 按 batch 对 ConvBlock 中内容 和 
    /*int ImageSize3D = ImageSize2D * CK->ConvKernelCount;
    for (int i = 0; i < BatchSize; i++)
    {
        Conv2DPerImage(CB, ConvBlocksPerBatch, i, CK, &ImageOutData[i * ImageSize3D], ImageSize2D, &ReluOutData[i * ImageSize3D]);
    }*/
}

void SdlfFunctionCUDA::Conv2DPerImage(ConvBlocks* CB, int ConvBlocksPerBatch, int BatchIndex, ConvKernel* CK, float* ImageOutData, int ImageSize2D, float* ReluOutData)
{
    // 用每一卷积核 对 图片进行卷积
    for (int i = 0; i < CK->ConvKernelCount; i++)
    {
        Conv2DPerImagePerConvKernel(CB, ConvBlocksPerBatch, BatchIndex, CK, i, &ImageOutData[ImageSize2D * i], &ReluOutData[ImageSize2D * i]);
    }
}

void SdlfFunctionCUDA::Conv2DPerImagePerConvKernel(ConvBlocks* CB, int ConvBlocksPerBatch, int BatchIndex, ConvKernel* CK, int ConvKernelIndex, float* ImageOutData, float* ReluOutData)
{
    // ConvBlocksPerBatch 为 CB 中包含的每 batch 的 ConvLen 长度 array 个数
    int StartPos = ConvBlocksPerBatch * BatchIndex;
    for (int i = 0; i < ConvBlocksPerBatch; i++)
    {
        float ConvResult = CK->Conv2D(&CB->ConvArray[(StartPos + i) * CB->ArrayLen], ConvKernelIndex);
        if (ConvResult > 0)
        {
            // 假定ImageOutData和ReluOutData已经初始化为0
            ImageOutData[i] = ConvResult;
            ReluOutData[i] = 1.0f;
        }
    }
}

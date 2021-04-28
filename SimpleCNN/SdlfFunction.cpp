/***********************************************
 * File: SdlfFunction.cpp
 *
 * Author: CHB
 * Date: 2020-04-18
 *
 * Purpose:���峣�ú���
 *
 *
 **********************************************/
#include <math.h>
#include <algorithm>
#include "SdlfFunction.h"
#include "KernelCPU.h"
#include "Common.h"

float SdlfFunction::MaxInPool(float* Data, int Num, int& MaxIndex)
{
    // ��ð��һ��
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

float SdlfFunction::DotProduct(float* Data1, float* Data2, int Num)
{
    float Sum = 0.0f;
    for (int i = 0; i < Num; i++)
    {
        Sum += Data1[i] * Data2[i];
    }

    return Sum;
}

float SdlfFunction::UnitRandom()
{
    static bool FirstRun = true;
    if (FirstRun) {
        srand((unsigned)time(NULL));
        FirstRun = false;
    }
    return float(rand()) / float(RAND_MAX);
}

float SdlfFunction::RangeRandom(float Min, float Max)
{
    return Min + (Max - Min) * UnitRandom();
}

double SdlfFunction::Pow(double x, double y)
{
    return pow(x, y);
}

double SdlfFunction::Exp(double n)
{
    return exp(n);
}

double SdlfFunction::ln(double n)
{
    return log(n);
}

double SdlfFunction::lg(double n)
{
    return log10(n);
}

double SdlfFunction::log_m_n(double m, double n)
{
    return ln(n) / ln(m);
}

void SdlfFunction::softmax(float* InArrayDest, float* OutArray, int Num)
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


void SdlfFunction::BuildConvArrayPerBatch(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int ConvWidth, int ConvHeight, int BatchSizeIndex)
{
    // ÿ��С��Ĵ�С���Ǿ���˵Ĵ�С
    int ConvLen = ConvWidth * ConvHeight * ImageDepth;
    // ����ͼƬ��һ��һ�������С��
    // ÿ��Batch����ô���С��
    int ImgSizePerChannel = ImageHeight * ImageWidth;
    int HalfConv = ConvWidth >> 1;
    float* ConvQuad = new float[ConvLen];
    for (int i = 0; i < ImageHeight; i++) {
        for (int j = 0; j < ImageWidth; j++) {
            // ��ʼ���quad
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

void SdlfFunction::BuildConvArray(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int ConvWidth, int ConvHeight, int BatchSize)
{
    int ConvLen = ConvWidth * ConvHeight * ImageDepth;
    int ImageSize = ImageWidth * ImageHeight * ImageDepth;
    if (CB->ConvArray == nullptr) {
        CB->ConvArray = new float[ImageWidth * ImageHeight * BatchSize * ConvLen]; 
        CB->ArrayLen = ConvLen;
    }
    // �� batch ���� convArray ���м���
    for (int i = 0; i < BatchSize; i++) {
        int Pos = i * ImageSize;
        BuildConvArrayPerBatch(CB, &ImageData[Pos], ImageWidth, ImageHeight, ImageDepth, ConvWidth, ConvHeight, i);
    }
    return;
}

void SdlfFunction::BuildFullConnectedArrayPerBatch(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int BatchIndex)
{
    int ImgSize = ImageWidth * ImageHeight * ImageDepth;
    memcpy(CB->ConvArray + BatchIndex * ImgSize, ImageData, sizeof(float) * ImgSize);
}

void SdlfFunction::BuildFullConnectedArray(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int BatchSize)
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

void SdlfFunction::ConvFullConnected(ConvBlocks* CB, ConvKernel* CK, float* ImageOutData, float* ReluOutData, int BatchSize)
{
    // ����ȫ������˵��һ��ͼƬ����һ���������������ľ����������batch��
    for (int i = 0; i < BatchSize; i++)
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
    }
}

void SdlfFunction::Conv2D(ConvBlocks* CB, int ConvBlocksPerBatch, ConvKernel* CK, float* ImageOutData, int ImageSize2D, float* ReluOutData, int BatchSize)
{
    // �� batch �� ConvBlock ������ �� 
    int ImageSize3D = ImageSize2D * CK->ConvKernelCount;
    for (int i = 0; i < BatchSize; i++)
    {
        Conv2DPerImage(CB, ConvBlocksPerBatch, i, CK, &ImageOutData[i * ImageSize3D], ImageSize2D, &ReluOutData[i * ImageSize3D]);
    }
}

void SdlfFunction::Conv2DPerImage(ConvBlocks* CB, int ConvBlocksPerBatch, int BatchIndex, ConvKernel* CK, float* ImageOutData, int ImageSize2D, float* ReluOutData)
{
    // ��ÿһ����� �� ͼƬ���о��
    for (int i = 0; i < CK->ConvKernelCount; i++)
    {
        Conv2DPerImagePerConvKernel(CB, ConvBlocksPerBatch, BatchIndex, CK, i, &ImageOutData[ImageSize2D * i], &ReluOutData[ImageSize2D * i]);
    }
}

void SdlfFunction::Conv2DPerImagePerConvKernel(ConvBlocks* CB, int ConvBlocksPerBatch, int BatchIndex, ConvKernel* CK, int ConvKernelIndex, float* ImageOutData, float* ReluOutData)
{
    // ConvBlocksPerBatch Ϊ CB �а�����ÿ batch �� ConvLen ���� array ����
    int StartPos = ConvBlocksPerBatch * BatchIndex;
    for (int i = 0; i < ConvBlocksPerBatch; i++)
    {
        float ConvResult = CK->Conv2D(&CB->ConvArray[(StartPos + i) * CB->ArrayLen], ConvKernelIndex);
        if (ConvResult > 0)
        {
            // �ٶ�ImageOutData��ReluOutData�Ѿ���ʼ��Ϊ0
            ImageOutData[i] = ConvResult;
            ReluOutData[i] = 1.0f;
        }
    }
}



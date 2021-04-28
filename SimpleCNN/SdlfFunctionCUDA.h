/***********************************************
 * File: SdlfFunction.h
 *
 * Author: CHB
 * Date: 2020-04-08
 *
 * Purpose:
 *
 *
 **********************************************/
#pragma once
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct ConvBlocks;
struct ConvKernel;
struct ConvBlocksCUDA;
struct ConvKernelCUDA;
// 把常用的函数封装在这里
class SdlfFunctionCUDA
{
public:
	SdlfFunctionCUDA() {};
	~SdlfFunctionCUDA() {};

	// 求最大值，主要是池化会用到？
	static float MaxInPool(float* Data, int Num, int& MaxIndex);

	// 点积
	static float DotProduct(float* Data1, float* Data2, int Num);
	// 单位随机数
	static float UnitRandom();
	// 范围随机数
	static float RangeRandom(float Min, float Max);
	// x^y
	static double Pow(double x, double y);
	// e^n
	static double Exp(double n);
	// ln(n)，e为底数的对数函数
	static double ln(double n);
	// lg(n)，10为底数的对数函数
	static double lg(double n);
	// log(m,n)，m为底数的对数函数。有时候要用到这种，例如交叉熵，有用e为底数，但是我看有些地方是2为底数
	static double log_m_n(double m, double n);

	// softmax，分类问题里面非常常用的函数
	// 这里用float作为输入，是因为使用GPU计算的时候，全部都是float。这里使用double没什么意义。
	// InArray和OutArray必须有一样的Num
	// InArrayDest：目标结果，InArrayDest：实际计算结果。
	static void softmax(float* InArrayDest, float* OutArray, int Num);
	// CrossEntropy，交叉熵。
	static double CrossEntropy(float* InArray, int* InArrayDest, int Num);



	// 这里不输入卷积核的depth，这个必须跟输入image的depth一致
	static void BuildConvArrayPerBatch(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int ConvWidth, int ConvHeight, int BatchSizeIndex);
	static void BuildConvArray(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int ConvWidth, int ConvHeight, int BatchSize);
	// 这个其实很简单，但是为了跟卷积核对齐，所以用了一样的方式
	static void BuildFullConnectedArrayPerBatch(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int BatchIndex);
	static void BuildFullConnectedArray(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int BatchSize);
	// 计算全连接。
	static void ConvFullConnected(ConvBlocks* CB, ConvKernel* CK, float* ImageOutData, float* ReluOutData, int BatchSize);

	// 这里有点绕，一张图片，需要跟N个卷积核做卷积，生成N张图片。这里是一张图片做卷积，结果也是一张图片。
	// 为什么绕，因为涉及到batch，涉及到多个卷积核。而batch再多，卷积核数量是不变的
	static void Conv2D(ConvBlocks* CB, int ConvBlocksPerBatch, ConvKernel* CK, float* ImageOutData, int ImageSize2D, float* ReluOutData, int BatchSize);
	static void Conv2DPerImage(ConvBlocks* CB, int ConvBlocksPerBatch, int BatchIndex, ConvKernel* CK, float* ImageOutData, int ImageSize2D, float* ReluOutData);
	static void Conv2DPerImagePerConvKernel(ConvBlocks* CB, int ConvBlocksPerBatch, int BatchIndex, ConvKernel* CK, int ConvKernelIndex, float* ImageOutData, float* ReluOutData);

};

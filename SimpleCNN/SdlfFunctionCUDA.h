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
// �ѳ��õĺ�����װ������
class SdlfFunctionCUDA
{
public:
	SdlfFunctionCUDA() {};
	~SdlfFunctionCUDA() {};

	// �����ֵ����Ҫ�ǳػ����õ���
	static float MaxInPool(float* Data, int Num, int& MaxIndex);

	// ���
	static float DotProduct(float* Data1, float* Data2, int Num);
	// ��λ�����
	static float UnitRandom();
	// ��Χ�����
	static float RangeRandom(float Min, float Max);
	// x^y
	static double Pow(double x, double y);
	// e^n
	static double Exp(double n);
	// ln(n)��eΪ�����Ķ�������
	static double ln(double n);
	// lg(n)��10Ϊ�����Ķ�������
	static double lg(double n);
	// log(m,n)��mΪ�����Ķ�����������ʱ��Ҫ�õ����֣����罻���أ�����eΪ�����������ҿ���Щ�ط���2Ϊ����
	static double log_m_n(double m, double n);

	// softmax��������������ǳ����õĺ���
	// ������float��Ϊ���룬����Ϊʹ��GPU�����ʱ��ȫ������float������ʹ��doubleûʲô���塣
	// InArray��OutArray������һ����Num
	// InArrayDest��Ŀ������InArrayDest��ʵ�ʼ�������
	static void softmax(float* InArrayDest, float* OutArray, int Num);
	// CrossEntropy�������ء�
	static double CrossEntropy(float* InArray, int* InArrayDest, int Num);



	// ���ﲻ�������˵�depth��������������image��depthһ��
	static void BuildConvArrayPerBatch(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int ConvWidth, int ConvHeight, int BatchSizeIndex);
	static void BuildConvArray(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int ConvWidth, int ConvHeight, int BatchSize);
	// �����ʵ�ܼ򵥣�����Ϊ�˸�����˶��룬��������һ���ķ�ʽ
	static void BuildFullConnectedArrayPerBatch(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int BatchIndex);
	static void BuildFullConnectedArray(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int BatchSize);
	// ����ȫ���ӡ�
	static void ConvFullConnected(ConvBlocks* CB, ConvKernel* CK, float* ImageOutData, float* ReluOutData, int BatchSize);

	// �����е��ƣ�һ��ͼƬ����Ҫ��N������������������N��ͼƬ��������һ��ͼƬ����������Ҳ��һ��ͼƬ��
	// Ϊʲô�ƣ���Ϊ�漰��batch���漰���������ˡ���batch�ٶ࣬����������ǲ����
	static void Conv2D(ConvBlocks* CB, int ConvBlocksPerBatch, ConvKernel* CK, float* ImageOutData, int ImageSize2D, float* ReluOutData, int BatchSize);
	static void Conv2DPerImage(ConvBlocks* CB, int ConvBlocksPerBatch, int BatchIndex, ConvKernel* CK, float* ImageOutData, int ImageSize2D, float* ReluOutData);
	static void Conv2DPerImagePerConvKernel(ConvBlocks* CB, int ConvBlocksPerBatch, int BatchIndex, ConvKernel* CK, int ConvKernelIndex, float* ImageOutData, float* ReluOutData);

};

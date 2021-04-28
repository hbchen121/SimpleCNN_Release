/***********************************************
 * File: SdlfCalculator.h
 *
 * Author: CHB
 * Date: 2021-04-07
 *
 * Purpose:�Ѵ����ļ��㼯����һ���ط�����һ������
 * �ײ������CPU��Compute Shader, Cuda, Cudnn������ʱֻ����ʵ��CPU��Compute Shader��
 * �����ϣ�CPUЧ����ͣ����������׿�����㷨��Compute Shader��Cuda�����ƣ���������ò���Ǹ��������Ȳ�ͬ������CS�����Կ���ʲô�Կ����У�Cuda����N����
 * Cudnn��򵥣���ΪCudnn����Ӧ�ü�����һ����ֳ��㷨��ֱ�ӵ��ü��ɡ���ϣ���Լ�ʵ��һ��ײ��㷨�������Լ�����⡣
 **********************************************/

#pragma once
#include "Common.h"
#include "Sdlf.h"

enum SdlfCalculatorType;
enum SdlfActivationFunc;

struct ConvKernel;
struct SoftMaxKernel;

class SdlfCalculator
{
public:
	static SdlfCalculator* CreateCalculator(SdlfCalculatorType SCT);
	// 2D ���
	virtual void Conv2D(float* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchCount, ConvKernel* CK, SdlfActivationFunc ActivationFunc) = 0;
	virtual void Conv2D(ConvKernel* CK, SdlfActivationFunc ActivationFunc) = 0;

	//virtual void Conv2D_2(float* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchCount, ConvKernel* CK, SdlfActivationFunc ActivationFunc) = 0;
	//virtual void Conv2D_2(ConvKernel* CK, SdlfActivationFunc ActivationFunc) = 0;

	// ��ȡ�ϴ��������Ϊ�ϴ�������������Դ���ģ����ʱ����ȫ����Ҫ���¶����ڴ棬ֱ�ӻ�ȡһ��ָ�뼴��
	// virtual void* GetLastOutput(int& Width, int& Height, int& Depth, int& BatchCount) = 0;
	
	// �ػ����㣬�ڶ�����������֪����������������GPUʵ�֣��ײ�ֱ��ʵ�֣�����Ҫ���봫������
	virtual void Max_Pool_2_2(int InWidth, int InHeight, float* InData, float* MaxPoolGradientData, float* OutData) = 0;
	virtual void Max_Pool_2_2(ConvKernel* CK) = 0;
	
	// ͼƬ��ʽת�������ǵ�batch��������GPUת�������ʱ�����õ�һ��ͼƬ����
	virtual void Transform_uchar_to_float(unsigned char* InData, float* OutData, int DataLen) = 0;
	
	// ȫ���Ӳ����
	virtual void CalcFullLink(ConvKernel* FLinkKernel) = 0;
	virtual void CalcFullLink(float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, ConvKernel* FLinkKernel) = 0;

	// softmax
	virtual float* SoftMax(SoftMaxKernel* SMK) = 0;
	virtual float* SoftMax() = 0;
	virtual void Release() = 0;

protected:
	SdlfCalculator()
	{
		mInWidth = 0;
		mInHeight = 0;
		mInChannel = 0;
		mOutWidth = 0;
		mOutHeight = 0;
		mOutChannel = 0;
		mPadding = 0;
		mBatchSize = 32;
	}
	virtual ~SdlfCalculator() {};

	static float UCharToFloat(unsigned char C);
	// static float UCharToFloat_0_1(unsigned char C);
	// static unsigned char FloatToUChar(float F);
	// static unsigned char Float_0_1_ToUChar(float F);

	int mInWidth;
	int mInHeight;
	int mInChannel;
	int mOutWidth;
	int mOutHeight;
	int mOutChannel;

	int mPadding;
	int mBatchSize;
};

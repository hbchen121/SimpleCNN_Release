/***********************************************
 * File: SdlfCalculatorCPU.h
 *
 * Author: CHB
 * Date: 2020-04-08
 *
 * Purpose:
 *
 *
 **********************************************/
#pragma once

#include "SdlfCalculator.h"
#include "KernelCPU.h"

class SdlfCalculatorCPU : public SdlfCalculator
{
	friend class SdlfCalculator;
public:
	// 2d conv
	virtual void Conv2D(float* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchSize, ConvKernel* CK, SdlfActivationFunc ActivationFunc) final;
	virtual void Conv2D(ConvKernel* CK, SdlfActivationFunc ActivationFunc) final;

	virtual void Transform_uchar_to_float(unsigned char* InData, float* OutData, int DataLen) final;

	// 池化计算
	virtual void Max_Pool_2_2(int InWidth, int InHeight, float* InData, float* MaxPoolGradientData, float* OutData) final;
	virtual void Max_Pool_2_2(ConvKernel* CK) final;

	// 全连接层计算
	virtual void CalcFullLink(ConvKernel* FLinkKernel) final;
	virtual void CalcFullLink(float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, ConvKernel* FLinkKernel) final;

	virtual float* SoftMax(SoftMaxKernel* SMK) final;
	virtual float* SoftMax() final;
	virtual void Release() final;
protected:
	SdlfCalculatorCPU();
	virtual ~SdlfCalculatorCPU();
private:
	// 图片输出信息。
	float* mImageOutData;
};

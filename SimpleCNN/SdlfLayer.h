/***********************************************
 * File: SdlfLayer.h
 *
 * Author: CHB
 * Date: 2021-04-07
 *
 * Purpose:
 *
 *
 **********************************************/
#pragma once

#include "Common.h"
#include "KernelCPU.h"
#include "KernelCUDA.h"
#include "Sdlf.h"
#include "SdlfCalculator.h"

class SdlfLayerImpl : public SdlfLayer
{
public:
	SdlfLayerImpl();
	virtual ~SdlfLayerImpl();
	virtual void SetLayerType(SdlfLayerType LT) override;
	virtual SdlfLayerType GetLayerType() const override;

	virtual SdlfLayer* GetPreLayer() override;
	virtual void SetPreLayer(SdlfLayer* Layer)override;
	virtual SdlfLayer* GetNextLayer() override;
	virtual void SetNextLayer(SdlfLayer* Layer) override;

	// Must be convolution layer
	virtual bool SetConvKernel(int ConvKernelWidth, int ConvKernelHeight, int InWidth, int InHeight, int InChannels, int OutChannels) override;
	virtual bool SetConvKernelCUDA() override;
	// ��ʼ������˵Ĳ��������Ĭ��������ġ���������Ҫ�Ļ����ҿ�Ҳ�г�ʼ���ģ�
	// ֻ����w��b
	virtual void SetConvParam(float W, float B) override;
	virtual void InitializeParamToRandom(float fMin, float fMax) override;
	virtual bool SetActivationFunc(SdlfActivationFunc AF) override;
	virtual bool SetPoolParam(PoolingType PT) override;
	// ȫ���Ӳ�������á���˵���ںܶ���㷨�Ѿ�������ȫ���Ӳ㣬���õ�ȫ����ƽ��ֵ��
	// ����Ҳ����һ�£���һ����ֵ�Ĳ���һ��Ч����������û��ʲô��ͬ
	virtual bool SetFullLinkParam(int Width, int Height, int Depth, int FullLinkDepth) override;


	// �����󵼲�����Ŀǰ������һ��step��
	virtual void SetGradientParam(float Step) override;
	virtual void Release() override;

	float* Excute(float* ImageData, int BatchSize, SdlfCalculator* Calculator);
	// ��Calculator��ȡ��һ�ε��������Ϊ��ε������������GPU�ˣ���ȫû��Ҫ��ȡ������
	// ���ط��������������������󵼻����Ǽ���׼ȷ��
	float* Excute(SdlfCalculator* Calculator);
	// ������
	void SoftMaxGradient(float* GradientData, int BatchSize, SdlfCalculator* Calculator);
	void FullLinkGradient(float* GradientData, int BatchSize, SdlfCalculator* Calculator);
	void Conv2DGradient(float* GradientData, int BatchSize, SdlfCalculator* Calculator);

	void UpdateStep(float Multi);
protected:
	ConvKernel* GetConvKernel();
	ConvKernel* GetFullLinkKernel();
private:
	SdlfLayerType mLayerType;
	SdlfActivationFunc mActivationFunc;
	PoolingType mPoolType;
	// ��¼һЩ����������Ҫ����һ��padding�����Ǽ����Զ������������ÿһ�ξ��������ı��ߡ����Ǽ���ͼƬ���һ��
	int mInWidth;
	int mInHeight;
	int mInChannel;
	int mPadding;


	SdlfLayerImpl* mPreLayer;
	SdlfLayerImpl* mNextLayer;

	float mStep;
	// �����
	ConvKernel* mConvKernel;
	ConvKernelCUDA* mConvKernelCUDA;
	ConvKernelCUDA* d_mConvKernelCUDA;

	// ȫ���Ӳ�����Ԥ��Ҳ����Ҫ�󵼸��µ�
	ConvKernel* mFullLinkKernel;
	ConvKernelCUDA* mFullLinkKernelCUDA;
	ConvKernelCUDA* d_mFullLinkKernelCUDA;
	// SoftMax����
	SoftMaxKernel* mSoftMaxKernel;
	SoftMaxKernelCUDA* mSoftMaxKernelCUDA;
	SoftMaxKernelCUDA* d_mSoftMaxKernelCUDA;
};

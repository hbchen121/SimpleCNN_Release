/***********************************************
 * File: SdlfLayer.cpp
 *
 * Author: CHB
 * Date: 2020-04-18
 *
 * Purpose:
 *
 *
 **********************************************/
#include "Common.h"
#include "SdlfLayer.h"

SdlfLayerImpl::SdlfLayerImpl()
{
	mLayerType = Convolution;
	mActivationFunc = Relu;
	mPoolType = Max_Pooling;
	mInWidth = 0;
	mInHeight = 0;
	mInChannel = 0;
	mPadding = 0;
	mStep = STEP;
	mPreLayer = nullptr;
	mNextLayer = nullptr;
	mConvKernel = nullptr;
	mFullLinkKernel = nullptr;
	mSoftMaxKernel = nullptr;
}

SdlfLayerImpl::~SdlfLayerImpl()
{
	SAFE_DELETE(mConvKernel);
	SAFE_DELETE(mFullLinkKernel);
	SAFE_DELETE(mSoftMaxKernel);
}

void SdlfLayerImpl::SetLayerType(SdlfLayerType LT)
{
	mLayerType = LT;
}

SdlfLayerType SdlfLayerImpl::GetLayerType() const
{
	return SdlfLayerType();
}

SdlfLayer* SdlfLayerImpl::GetPreLayer()
{
	return nullptr;
}

void SdlfLayerImpl::SetPreLayer(SdlfLayer* Layer)
{
	mPreLayer = static_cast<SdlfLayerImpl*>(Layer);
}

SdlfLayer* SdlfLayerImpl::GetNextLayer()
{
	return nullptr;
}

void SdlfLayerImpl::SetNextLayer(SdlfLayer* Layer)
{
	mNextLayer = static_cast<SdlfLayerImpl*>(Layer);
}

bool SdlfLayerImpl::SetConvKernel(int ConvKernelWidth, int ConvKernelHeight, int InWidth, int InHeight, int InChannels, int OutChannels)
{
	mInWidth = InWidth;
	mInHeight = InHeight;
	mInChannel = InChannels;
	mPadding = (ConvKernelWidth - 1) >> 1;  // ���� ConvKernel Size �� n*n ��
	mConvKernel = new ConvKernel(ConvKernelWidth, ConvKernelHeight, InChannels, OutChannels);
	
	return true;
}

bool SdlfLayerImpl::SetConvKernelCUDA()
{
	mConvKernelCUDA = new ConvKernelCUDA(mConvKernel);
	cudaError_t Err;
	Err = CUDA_MALLOC(d_mConvKernelCUDA, sizeof(ConvKernelCUDA));
	HANDLE_ERROR(Err);
	Err = cudaMemcpy(d_mConvKernelCUDA, mConvKernelCUDA, sizeof(ConvKernelCUDA), cudaMemcpyHostToDevice);
	HANDLE_ERROR(Err);
	return false;
}

void SdlfLayerImpl::SetConvParam(float W, float B)
{
	switch (mLayerType)
	{
	case Convolution:
	{
		if (mConvKernel)
			mConvKernel->InitializeData(W, B);
		
		break;
	}
	case FullyConnected:
	{
		if (mFullLinkKernel)
			mFullLinkKernel->InitializeData(W, B);
		break;
	}
	case SoftMax:
	{
		if (!mSoftMaxKernel)
		{
			mSoftMaxKernel = new SoftMaxKernel(1024, 10);
		}
		mSoftMaxKernel->InitializeData(W, B);
	break;
	}
	}
}

void SdlfLayerImpl::InitializeParamToRandom(float fMin, float fMax)
{
	switch (mLayerType)
	{
	case Convolution:
	{
		if (mConvKernel) {
			mConvKernel->InitializeDataToRandom(fMin, fMax);
		}
		break;
	}
	case FullyConnected:
	{
		if (mFullLinkKernel) {
			mFullLinkKernel->InitializeDataToRandom(fMin, fMax);
		}
		break;
	}
	case FeatureMapAvg:
		break;
	case SoftMax:
	{
		if (!mSoftMaxKernel)
		{
			mSoftMaxKernel = new SoftMaxKernel(1024, 10);
		}
		mSoftMaxKernel->InitializeDataToRandom(fMin, fMax);
		break;
	}
		
	default:
		break;
	};
}

bool SdlfLayerImpl::SetActivationFunc(SdlfActivationFunc AF)
{
	mActivationFunc = AF;
	return true;
}

bool SdlfLayerImpl::SetPoolParam(PoolingType PT)
{
	mPoolType = PT;
	return true;
}

bool SdlfLayerImpl::SetFullLinkParam(int Width, int Height, int Depth, int FullLinkDepth)
{
	SAFE_DELETE(mFullLinkKernel);
	mFullLinkKernel = new ConvKernel(Width, Height, Depth, FullLinkDepth);
	mInChannel = Depth;
	mInWidth = Width;
	mInHeight = Height;
	return true;
}

void SdlfLayerImpl::SetGradientParam(float Step)
{
	mStep = Step;
}

void SdlfLayerImpl::Release()
{
	delete this;
}

float* SdlfLayerImpl::Excute(float* ImageData, int BatchSize, SdlfCalculator* Calculator)
{
	switch (mLayerType)
	{
	case Convolution:
	{
		Calculator->Conv2D_2(ImageData, mInWidth, mInHeight, mInChannel, BatchSize, mConvKernel, mActivationFunc);
		Calculator->Max_Pool_2_2(mConvKernel);
		return mNextLayer->Excute(Calculator);
	}
	case FullyConnected:
	{
		Calculator->CalcFullLink(ImageData, mInWidth, mInHeight, mInChannel, mFullLinkKernel);
		return mNextLayer->Excute(Calculator);
	}
	case FeatureMapAvg:
		break;
	case SoftMax:
	{
		break;
	}
	default:
		break;
	}
	return nullptr;
}

float* SdlfLayerImpl::Excute(SdlfCalculator* Calculator)
{
	switch (mLayerType)
	{
	case Convolution:
	{
		Calculator->Conv2D_2(mConvKernel, mActivationFunc);
		Calculator->Max_Pool_2_2(mConvKernel);
		return mNextLayer->Excute(Calculator);
	}
	case FullyConnected:
	{
		Calculator->CalcFullLink(mFullLinkKernel);
		return mNextLayer->Excute(Calculator);
	}
	case FeatureMapAvg:
		break;
	case SoftMax:
	{
		float* Result = Calculator->SoftMax(mSoftMaxKernel);
		// float* Result = Calculator->SoftMax();
		// std::cout << (float)Result[0] << std::endl;
		return Result;
	}
	default:
		break;
	}
	return nullptr;
}

void SdlfLayerImpl::SoftMaxGradient(float* GradientData, int BatchSize, SdlfCalculator* Calculator)
{
	// �ȼ�����ǰ�������ٸ��²���
	float* Output = new float[BatchSize * mSoftMaxKernel->Row];

	mSoftMaxKernel->CalcGradient(BatchSize, GradientData, Output);
	// ���²���
	mSoftMaxKernel->UpdateParameter(GradientData, BatchSize, mStep);
	// ���ϴ���
	mPreLayer->FullLinkGradient(Output, BatchSize, Calculator);

	SAFE_DELETE_ARRAY(Output);
	return;
}

void SdlfLayerImpl::FullLinkGradient(float* GradientData, int BatchCount, SdlfCalculator* Calculator)
{
	mFullLinkKernel->CalcFullLinkDropOutAndReluGradient(GradientData, BatchCount);

	// ������ǰȫ������
	float* OutData = new float[BatchCount * mFullLinkKernel->ConvKernelWidth * mFullLinkKernel->ConvKernelHeight * mFullLinkKernel->ConvKernelChannel];

	mFullLinkKernel->CalcFullLinkGradient(GradientData, BatchCount, OutData);
	// ����ȫ���Ӳ���
	mFullLinkKernel->UpdateFullLinkParameter(GradientData, BatchCount, mStep);
	// ������ǰ�����
	if (mPreLayer)
		mPreLayer->Conv2DGradient(OutData, BatchCount, Calculator);

	SAFE_DELETE_ARRAY(OutData);
}

void SdlfLayerImpl::Conv2DGradient(float* GradientData, int BatchCount, SdlfCalculator* Calculator)
{
	// GradientData�ĳ������ϴ�MaxPool��w * h * d * BatchCount
	// �����һ�ξ������14 * 14 * 32 * BatchCount���ڶ��ξ����7 * 7 * 64 * BatchCount
	// OutData�ĳ��ȣ���һ����28 * 28 * 32 * BatchCount���ڶ�����14 * 14 * 64 * BatchCount
	float* OutData = new float[mConvKernel->ImageInputWidth * mConvKernel->ImageInputHeight * mConvKernel->ConvKernelCount * BatchCount];
	// ������MaxPool��Relu
	mConvKernel->CalcConv2DMaxPoolAndReluGradient(GradientData, OutData, BatchCount);
	// ��������
	if (mPreLayer)		// ����������жϣ������ǵ�һ���ڵ㡣��һ���ڵ㣬����Ҫ����ǰ��
	{
		mConvKernel->ApplyConvKernelAndRotate180();
		// ������ǰ��
		int Size = mConvKernel->ImageInputWidth * mConvKernel->ImageInputHeight * mConvKernel->ConvKernelChannel * BatchCount;
		float* ConvGradientData = new float[Size];
		memset(ConvGradientData, 0, sizeof(float) * Size);

		mConvKernel->CalcConv2DGradient(OutData, BatchCount, ConvGradientData);
		// �����󵼲���
		mConvKernel->UpdateConv2DParameter(OutData, BatchCount, mStep);
		// ������ǰִ��
		mPreLayer->Conv2DGradient(ConvGradientData, BatchCount, Calculator);

		SAFE_DELETE_ARRAY(ConvGradientData);
	}
	else
	{
		// �����󵼲���
		mConvKernel->UpdateConv2DParameter(OutData, BatchCount, mStep);
	}

	SAFE_DELETE_ARRAY(OutData);
}

void SdlfLayerImpl::UpdateStep(float Multi)
{
	mStep *= Multi;
	if (mNextLayer)
	{
		mNextLayer->UpdateStep(Multi);
	}
}

ConvKernel* SdlfLayerImpl::GetConvKernel()
{
	return nullptr;
}

ConvKernel* SdlfLayerImpl::GetFullLinkKernel()
{
	return nullptr;
}

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
	mPadding = (ConvKernelWidth - 1) >> 1;  // 假设 ConvKernel Size 是 n*n 的
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
	// 先计算向前导数，再更新参数
	float* Output = new float[BatchSize * mSoftMaxKernel->Row];

	mSoftMaxKernel->CalcGradient(BatchSize, GradientData, Output);
	// 更新参数
	mSoftMaxKernel->UpdateParameter(GradientData, BatchSize, mStep);
	// 往上传递
	mPreLayer->FullLinkGradient(Output, BatchSize, Calculator);

	SAFE_DELETE_ARRAY(Output);
	return;
}

void SdlfLayerImpl::FullLinkGradient(float* GradientData, int BatchCount, SdlfCalculator* Calculator)
{
	mFullLinkKernel->CalcFullLinkDropOutAndReluGradient(GradientData, BatchCount);

	// 继续向前全连接求导
	float* OutData = new float[BatchCount * mFullLinkKernel->ConvKernelWidth * mFullLinkKernel->ConvKernelHeight * mFullLinkKernel->ConvKernelChannel];

	mFullLinkKernel->CalcFullLinkGradient(GradientData, BatchCount, OutData);
	// 更新全连接参数
	mFullLinkKernel->UpdateFullLinkParameter(GradientData, BatchCount, mStep);
	// 继续向前卷积求导
	if (mPreLayer)
		mPreLayer->Conv2DGradient(OutData, BatchCount, Calculator);

	SAFE_DELETE_ARRAY(OutData);
}

void SdlfLayerImpl::Conv2DGradient(float* GradientData, int BatchCount, SdlfCalculator* Calculator)
{
	// GradientData的长度是上次MaxPool的w * h * d * BatchCount
	// 例如第一次卷积，是14 * 14 * 32 * BatchCount；第二次卷积是7 * 7 * 64 * BatchCount
	// OutData的长度，第一次是28 * 28 * 32 * BatchCount；第二次是14 * 14 * 64 * BatchCount
	float* OutData = new float[mConvKernel->ImageInputWidth * mConvKernel->ImageInputHeight * mConvKernel->ConvKernelCount * BatchCount];
	// 反向求导MaxPool和Relu
	mConvKernel->CalcConv2DMaxPoolAndReluGradient(GradientData, OutData, BatchCount);
	// 反向卷积求导
	if (mPreLayer)		// 必须做这个判断，可能是第一个节点。第一个节点，不需要再往前算
	{
		mConvKernel->ApplyConvKernelAndRotate180();
		// 计算向前求导
		int Size = mConvKernel->ImageInputWidth * mConvKernel->ImageInputHeight * mConvKernel->ConvKernelChannel * BatchCount;
		float* ConvGradientData = new float[Size];
		memset(ConvGradientData, 0, sizeof(float) * Size);

		mConvKernel->CalcConv2DGradient(OutData, BatchCount, ConvGradientData);
		// 更新求导参数
		mConvKernel->UpdateConv2DParameter(OutData, BatchCount, mStep);
		// 继续向前执行
		mPreLayer->Conv2DGradient(ConvGradientData, BatchCount, Calculator);

		SAFE_DELETE_ARRAY(ConvGradientData);
	}
	else
	{
		// 更新求导参数
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

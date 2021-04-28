/***********************************************
 * File: SdlfModel.cpp
 *
 * Author: CHB
 * Date: 2021-04-07
 *
 * Purpose:
 *
 *
 **********************************************/
#include "SdlfModel.h"
#include "SdlfFunction.h"

SdlfModelImpl::SdlfModelImpl(SdlfCalculatorType SCT)
{
    mFirstLayer = nullptr;
    mLastLayer = nullptr;
    mImgWidth = 0;
    mImgHeight = 0;
    mImgChannel = 0;
    mTransformImgData = nullptr;
    mProcessType = ProcessTrain;
    mCorrectCount = 0;
    mInCorrectCount = 0;
    mAccuracyRate = 0.0f;
    mDynamicStep = false;
    SCT_Default = SCT;
    mCalculator = SdlfCalculator::CreateCalculator(SCT_Default);
}

SdlfModelImpl::~SdlfModelImpl()
{
    mCalculator->Release();
    mCalculator = nullptr;
    SAFE_DELETE_ARRAY(mTransformImgData);
}

void SdlfModelImpl::AddModelListener(SdlfModelListener* Listener)
{
    mListener.push_back(Listener);
    return;
}

void SdlfModelImpl::RemoveModelListener(SdlfModelListener* Listener)
{
    for (size_t i = 0; i < mListener.size(); i++)
    {
        if (mListener[i] == Listener)
        {
            mListener.erase(mListener.begin() + i);
            break;
        }
    }
    return;
}

void SdlfModelImpl::SetImgParam(int ImgWidth, int ImgHeight, int ImgChannel, int BatchSize)
{
    mImgWidth = ImgWidth;
    mImgHeight = ImgHeight;
    mImgChannel = ImgChannel;
    mBatchSize = BatchSize;

    mTransformImgData = new float[ImgWidth * ImgHeight * ImgChannel * mBatchSize];
}

void SdlfModelImpl::SetProcessType(ProcessType PT)
{
    mProcessType = PT;
}

void SdlfModelImpl::StartTrainSession(unsigned char* ImgData, unsigned char* Classification)
{
    if (mFirstLayer)
    {
        // �Ȱ�����ת��Ϊ float ����
        int Len = mImgWidth * mImgHeight * mImgChannel * mBatchSize;
        mCalculator->Transform_uchar_to_float(ImgData, mTransformImgData, Len);

        float* Result = mFirstLayer->Excute(mTransformImgData, mBatchSize, mCalculator);
		if (mProcessType == ProcessTrain)
		{
			// ִ����ɣ��󵼣����²���
			// �Ȱѷ�����ת��float�������ź���loss������
			// ��������loss������ƫ�����������������ܼ򵥣����������󵼹��̱Ƚϸ��ӣ�������ϸ�Ƶ���һ��ġ�������Կ����https://zhuanlan.zhihu.com/p/25723112

			float* ClassifyGrad = new float[10 * mBatchSize];
			float Errors = 0.0f;
			float Loss = 0.0f;
			for (int i = 0; i < mBatchSize * 10; i++)
			{
				// crossenropy �� y_i ���ݶ��� a_{y_i} - 1���������� a_j
				ClassifyGrad[i] = Result[i] - float(Classification[i]);
				if (Classification[i] == 1)
				{
					Loss += -SdlfFunction::ln(Result[i] + 1e-7);
				}
			}
			// ����׼ȷ�ʡ�����ÿ��ѵ������ȷ�ʣ�Ȼ����ͣ�����batch size
			for (int i = 0; i < mBatchSize; i++)
			{
				int MaxIndex = 0;
				SdlfFunction::MaxInPool(Result + i * 10, 10, MaxIndex);
				if (Classification[i * 10 + MaxIndex] == 1)
				{
					Errors += 1.0f;
				}
			}
			mLastLayer->SoftMaxGradient(ClassifyGrad, mBatchSize, mCalculator);
			SAFE_DELETE_ARRAY(ClassifyGrad);
			char Msg[256] = { 0 };
			Errors /= float(mBatchSize);
			Loss /= float(mBatchSize);
			if (mDynamicStep)
				UpdateStep(Loss);
			static int TrainTimes = 0;
			sprintf_s(Msg, 256, "Train Count: %d,Train AccuracyRate:%f, Loss:%f\n", TrainTimes++, Errors, Loss);
			NotifyMessage(Msg);
			if (fabs(Errors) >= 0.95f)
			{
				NotifyTrainComplete();
			}
		}
		else if (mProcessType == ProcessTest)
		{
			// ֱ�ӷ������ͽ������ʵ�ʽ�����գ�����׼ȷ�ʡ����⣬���ﲻ����batch��batch count����Ϊ1
			// ȡ���ֵ������������
			int MaxIndex = 0;
			SdlfFunction::MaxInPool(Result, 10, MaxIndex);
			if (Classification[MaxIndex] == 1)
			{
				mCorrectCount++;
			}
			else
			{
				mInCorrectCount++;
			}
			mAccuracyRate = float(mCorrectCount) / float(mCorrectCount + mInCorrectCount);
			// std::cout << mAccuracyRate << std::endl;
		}
	}
}

float SdlfModelImpl::GetAccuracyRate() const
{
	return mAccuracyRate;
}

SdlfLayer* SdlfModelImpl::CreateLayer(SdlfLayerType LT)
{
    SdlfLayerImpl* L = new SdlfLayerImpl;
    L->SetLayerType(LT);
    return L;
}

void SdlfModelImpl::SetFirstLayer(SdlfLayer* Layer)
{
    mFirstLayer = static_cast<SdlfLayerImpl*>(Layer);
}

void SdlfModelImpl::SetLastLayer(SdlfLayer* Layer)
{
    mLastLayer = static_cast<SdlfLayerImpl*>(Layer);
}

void SdlfModelImpl::SetDynamicStep(bool DynamicStep)
{
	mDynamicStep = DynamicStep;
}

void SdlfModelImpl::Release()
{
	delete this;
}

void SdlfModelImpl::NotifyMessage(const char* Msg)
{
	for (size_t i = 0; i < mListener.size(); i++)
	{
		mListener[i]->OnMessage(Msg);
	}
}

void SdlfModelImpl::NotifyTrainComplete()
{
	for (size_t i = 0; i < mListener.size(); i++)
	{
		mListener[i]->OnTrainComplete();
	}
}

void SdlfModelImpl::UpdateStep(float Loss)
{
	static bool UpdatePhase[3] = { false, false, false };
	if (Loss < 1.5f)
	{
		if (UpdatePhase[0] == false)
		{
			mFirstLayer->UpdateStep(0.1f);
			UpdatePhase[0] = true;
		}
	}
	if (Loss < 1.0f)
	{
		if (UpdatePhase[1] == false)
		{
			mFirstLayer->UpdateStep(0.5f);
			UpdatePhase[1] = true;
		}
	}
	if (Loss < 0.5f)
	{
		if (UpdatePhase[2] == false)
		{
			mFirstLayer->UpdateStep(0.5f);
			UpdatePhase[2] = true;
		}
	}
}

void DLL_CALLCONV CreateSdlfModel(SdlfModel** Model, SdlfCalculatorType SCT)
{
	SdlfModelImpl* M = new SdlfModelImpl(SCT);

	*Model = M;

	return;
}
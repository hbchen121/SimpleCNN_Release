/***********************************************
 * File: SdlfModel.h
 *
 * Author: CHB
 * Date: 2021-04-07
 *
 * Purpose:
 *
 *
 **********************************************/
#pragma once

#include <vector>
#include "Sdlf.h"
#include "SdlfLayer.h"
#include "SdlfCalculator.h"

class SdlfModelImpl : public SdlfModel
{
public:
	SdlfModelImpl(SdlfCalculatorType SCT=SCT_CPU);
	virtual ~SdlfModelImpl();

	virtual void AddModelListener(SdlfModelListener* Listener) override;
	virtual void RemoveModelListener(SdlfModelListener* Listener) override;

	// 设置
	virtual void SetImgParam(int ImgWidth, int ImgHeight, int ImgChannel, int BatchSize) override;
	virtual void SetProcessType(ProcessType PT) override;

	virtual void StartTrainSession(unsigned char* ImgData, unsigned char* Classification) override;
	virtual float GetAccuracyRate() const override;

	// Layer 是单向的，创建
	virtual SdlfLayer* CreateLayer(SdlfLayerType LT) override;
	virtual void SetFirstLayer(SdlfLayer* Layer) override;
	virtual void SetLastLayer(SdlfLayer* Layer) override;
	// 动态步长
	virtual void SetDynamicStep(bool DynamicStep) override;
	virtual void Release() override;

protected:
	static const int MaxBatchSize = 1024;
	float* mTransformImgData;
	void NotifyMessage(const char* Msg);
	void NotifyTrainComplete();
	void UpdateStep(float Loss);

private:
	ProcessType mProcessType;
	SdlfLayerImpl* mFirstLayer;
	SdlfLayerImpl* mLastLayer;
	std::vector<SdlfModelListener*> mListener;

	SdlfCalculator* mCalculator;
	SdlfCalculatorType SCT_Default;

	int mImgWidth;
	int mImgHeight;
	int mImgChannel;
	int mBatchSize;

	int mCorrectCount;
	int mInCorrectCount;
	float mAccuracyRate;
	bool mDynamicStep;
};

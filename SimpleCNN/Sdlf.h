/*
 * File: sdlf.h
 *
 * Author: CHB
 * Date: 2021-04-06
 *
 * Purpose:SDLF， short for simple deep learning framework.
 *
 *
 */
#pragma once
#define			API_EXPORT			extern "C" __declspec(dllexport)
#define			DLL_CALLCONV		__stdcall

enum SdlfCalculatorType
{
	SCT_CPU,
	SCT_CUDA,
};

enum SdlfActivationFunc
{
	ActiveNone,
	Sigmoid,
	Relu,
};

enum PoolingType
{
	Max_Pooling,
	Avg_Pooling,
};

enum SdlfLossFunc
{
	LogLoss,
	MSELoss,
	CrossEntorpyLoss,
};

enum SdlfLayerType
{
	Convolution,
	FullyConnected,
	FeatureMapAvg,
	SoftMax,
};

// 两种运行状态 训练-测试
enum ProcessType
{
	ProcessTrain,
	ProcessTest,
};

// Layer 控制相关
struct SdlfLayer
{
	virtual void SetLayerType(SdlfLayerType LT) = 0;
	virtual SdlfLayerType GetLayerType() const = 0;
	virtual SdlfLayer* GetPreLayer() = 0;
	virtual void SetPreLayer(SdlfLayer* Layer) = 0;
	virtual SdlfLayer* GetNextLayer() = 0;
	virtual void SetNextLayer(SdlfLayer* Layer) = 0;
	virtual bool SetConvKernel(int KernelWidth, int KernelHeight, int InWidth, int InHeight, int InChannels, int OutChannels) = 0;
	virtual bool SetConvKernelCUDA() = 0;
	virtual void SetConvParam(float W, float B) = 0;
	virtual void InitializeParamToRandom(float fWin, float fMax) = 0;
	virtual bool SetActivationFunc(SdlfActivationFunc AF) = 0;
	virtual bool SetPoolParam(PoolingType PT) = 0;
	virtual void SetGradientParam(float Step) = 0;
	// 特征全局求平均（舍弃全连接层）
	virtual bool SetFullLinkParam(int Width, int Height, int Depth, int FullLinkDepth) = 0;
	virtual void Release() = 0;

};

// 模型监听
struct SdlfModelListener
{
	virtual void OnMessage(const char* Msg) = 0;
	virtual void OnTrainComplete() = 0;
};

// 模型控制相关
struct SdlfModel
{
	// 模型监听
	virtual void AddModelListener(SdlfModelListener* Listener) = 0;
	virtual void RemoveModelListener(SdlfModelListener* Listener) = 0;

	// 设置
	virtual void SetImgParam(int ImgWidth, int ImgHeight, int ImgChannel, int BatchSize) = 0;
	virtual void SetProcessType(ProcessType PT) = 0;
	
	virtual void StartTrainSession(unsigned char* ImgData, unsigned char* Classification) = 0;
	virtual float GetAccuracyRate() const = 0;
	
	// Layer 是单向的，创建
	virtual SdlfLayer* CreateLayer(SdlfLayerType LT) = 0;
	virtual void SetFirstLayer(SdlfLayer* Layer) = 0;
	virtual void SetLastLayer(SdlfLayer* Layer) = 0;
	// 动态步长
	virtual void SetDynamicStep(bool DynamicStep) = 0;
	virtual void Release() = 0;
};


API_EXPORT void DLL_CALLCONV CreateSdlfModel(SdlfModel** Model, SdlfCalculatorType SCT=SCT_CPU);

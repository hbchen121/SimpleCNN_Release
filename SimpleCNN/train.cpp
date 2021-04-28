// train.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <vector>
#include <time.h>
#include "stdafx.h"
#include "Sdlf.h"
#include "MnistFileManager.h"

#define         BATCH_SIZE         1024
#define         DEFAULT_SCT           1
#define         TEST                  0
#define         TIMES                50
SdlfModel* g_Model = nullptr;
bool g_TrainCompleted = false;

std::vector<SdlfLayer* > g_LayerArray;

struct ModelListener : public SdlfModelListener
{
    virtual void OnMessage(const char* Msg) {
        printf("%s\n", Msg);
    }
    virtual void OnTrainComplete() {
        g_TrainCompleted = true;
    }
};

ModelListener Listener;

void ClearAll()
{
    for (size_t i = 0; i < g_LayerArray.size(); i++) {
        g_LayerArray[i]->Release();
    }
    g_LayerArray.clear();
    g_Model->Release();
    g_Model = nullptr;

    MnistFileManager::GetInstance()->Release();
}

void ModelSetup()
{
    g_Model->SetProcessType(ProcessTrain);
    g_Model->AddModelListener(&Listener);

    SdlfLayer* Layer1 = g_Model->CreateLayer(Convolution);
    g_Model->SetFirstLayer(Layer1);
    Layer1->SetActivationFunc(Relu);
    Layer1->SetConvKernel(5, 5, 28, 28, 1, 32);
    Layer1->InitializeParamToRandom(-0.01f, 0.01f);
    Layer1->SetGradientParam(0.01f);

    g_LayerArray.push_back(Layer1);

    SdlfLayer* Layer2 = g_Model->CreateLayer(FullyConnected);
    Layer2->SetActivationFunc(Relu);
    Layer2->SetFullLinkParam(14, 14, 32, 1024);
    Layer2->InitializeParamToRandom(-0.01f, 0.01f);
    Layer2->SetGradientParam(0.05f);

    Layer1->SetNextLayer(Layer2);
    Layer2->SetPreLayer(Layer1);
    g_LayerArray.push_back(Layer2);

    /*
    SdlfLayer* Layer3 = g_Model->CreateLayer(FullyConnected);
    Layer3->SetActivationFunc(Relu);
    Layer3->SetFullLinkParam(1, 1, 32, 1024);
    Layer3->InitializeParamToRandom(-0.01f, 0.01f);
    Layer3->SetGradientParam(0.05f);

    Layer2->SetNextLayer(Layer3);
    Layer3->SetPreLayer(Layer2);
    g_LayerArray.push_back(Layer3);
    */

    SdlfLayer* Layer4 = g_Model->CreateLayer(SoftMax);
    Layer4->InitializeParamToRandom(-0.01f, 0.01f);
    g_Model->SetLastLayer(Layer4);
    Layer2->SetNextLayer(Layer4);
    Layer4->SetPreLayer(Layer2);
    Layer4->SetGradientParam(0.05f);
    g_LayerArray.push_back(Layer4);

    g_Model->SetImgParam(28, 28, 1, BATCH_SIZE);
}

int main()
{
    SdlfCalculatorType SCT = SCT_CPU;
    if (DEFAULT_SCT == 1) {
        SCT = SCT_CUDA;
    }
    CreateSdlfModel(&g_Model, SCT);
    if(g_Model == nullptr) {
        printf("Create Sdlf model failed!\n");
        return 0;
    }
    if (MnistFileManager::GetInstance()->LoadTrainDataToMemory() == false) {
        printf("Load mnist file failed.\n");
        return 0;
    }
    ModelSetup();
    srand(time(NULL));
    // g_Model->SetProcessType(ProcessTest);
    
    double SumT = 0.;
    int t = 0;
    while (!g_TrainCompleted) {
        unsigned char* ImgData, * LabelData;
        MnistFileManager::GetInstance()->GetImageAndLabelDataInRandom(BATCH_SIZE, ImgData, LabelData);
        clock_t StartTime = clock();
        g_Model->StartTrainSession(ImgData, LabelData);
        clock_t EndTime = clock();
        double delta = ((double)EndTime - StartTime) / CLOCKS_PER_SEC;
        printf("%dth Comupting Time: %.5lf\n\n", t, delta);
        SumT += delta;
        if(++t == TIMES || TEST)    break;
    }
    printf("%d Times Comupting Time, Sum: %.5lf, Average: %.5lf\n\n", TIMES, SumT, SumT / TIMES);
    ClearAll();
    system("pause");
    return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件

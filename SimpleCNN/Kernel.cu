/***********************************************
 * File: Kernel.cu
 *
 * Author: CHB
 * Date: 2021-04-06
 *
 * Purpose:
 *		这个文件是 CUDA 的核心文件，所有cuda前向传播的过程
 *    都需要调用这个文件里的函数进行计算
 *
 **********************************************/
﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "KernelCUDA.h"
#include <assert.h>

#define APPLY_DATA_CUDA(V)														\
void Apply##V(float* Data, int Len)											\
{																			\
	if(V == nullptr)														\
	{																		\
		V = new float[Len];													\
	}																		\
	memcpy(V, Data, Len * sizeof(float));									\
}

void test_SM(SoftMaxKernel* SMK, int mBatchSize, float* mImageOutData)
{
    for (int i = 0; i < SMK->Row * SMK->Column; i++)
        SMK->W[i] = 1;
    for (int i = 0; i < SMK->Column; i++)
        SMK->B[i] = 1;
    for (int i = 0; i < mBatchSize * SMK->Row; i++)
        mImageOutData[i] = 1;
}

void test_CB(ConvBlocks* CB, ConvKernel* CK, int BatchSize, float* mImageOutData, int len)
{
    if(CB)
        for (int i = 0; i < CB->ArrayLen * BatchSize; i++)
            CB->ConvArray[i] = 1;
    if (CK) {
        int W_Size = CK->ConvKernelWidth * CK->ConvKernelHeight * CK->ConvKernelChannel * CK->ConvKernelCount;
        int B_Size = CK->ConvKernelCount;
        for (int i = 0; i < W_Size; i++)
            CK->W[i] = 1;
        for (int i = 0; i < B_Size; i++)
            CK->B[i] = 1;
    }
    if(mImageOutData)
        for (int i = 0; i < len; i++)
            mImageOutData[i] = 1;
}

__global__ void FullyConnected(ConvBlocksCUDA* CBC, ConvKernelCUDA* CKC, float* ImgOut, float* ReluOut, int BatchSize)
{
    int OutChannel = CKC->ConvKernelCount;
    int col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y * blockDim.y + threadIdx.y;
    int ImgSize = CKC->ConvKernelWidth * CKC->ConvKernelHeight * CKC->ConvKernelChannel;
    if (row < BatchSize && col < OutChannel) {
        float ConvRes = 0;
        for (int i = 0; i < ImgSize; i++) {
            ConvRes += CBC->ConvArray[row * ImgSize + i] * CKC->W[col * ImgSize + i];
        }
        ConvRes += CKC->B[col];
        if (ConvRes > 0.0f) {
            ImgOut[row * OutChannel + col] = ConvRes;
            ReluOut[row * OutChannel + col] = 1.0f;
        }
    }
}

extern "C" float* ConvFullConnectedCUDA(ConvBlocks* CB, ConvKernel* CK, float* ImageOutData, float* ReluOutData, int BatchSize)
{
    cudaError_t Err;
    // 对于全连接来说，一张图片生成一个卷积，所以这里的卷积数，就是batch数
    int imgSize = CB->ArrayLen, OutChannel = CK->ConvKernelCount;
    float* d_ConvArray, * d_imgOut, * d_reluOut;

    Err = CUDA_MALLOC(d_imgOut, BatchSize * OutChannel * sizeof(float));
    HANDLE_ERROR(Err);

    Err = CUDA_MALLOC(d_reluOut, BatchSize * OutChannel * sizeof(float));
    HANDLE_ERROR(Err);

    ConvBlocksCUDA* CBC = new ConvBlocksCUDA(CB, BatchSize, 1), * d_CBC;
    ConvKernelCUDA* CKC = new ConvKernelCUDA(CK), * d_CKC;
    Err = CUDA_MALLOC(d_CBC, sizeof(ConvBlocksCUDA));
    HANDLE_ERROR(Err);
    Err = cudaMemcpy(d_CBC, CBC, sizeof(ConvBlocksCUDA), cudaMemcpyHostToDevice);
    HANDLE_ERROR(Err);

    Err = CUDA_MALLOC(d_CKC, sizeof(ConvKernelCUDA));
    HANDLE_ERROR(Err);
    Err = cudaMemcpy(d_CKC, CKC, sizeof(ConvKernelCUDA), cudaMemcpyHostToDevice);
    HANDLE_ERROR(Err);

    dim3 Block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 Grid((OutChannel - 1 + BLOCK_SIZE) / BLOCK_SIZE, (BatchSize - 1 + BLOCK_SIZE) / BLOCK_SIZE);
    FullyConnected <<<Grid, Block>>> (d_CBC, d_CKC, d_imgOut, d_reluOut, BatchSize);
    float* OutData = nullptr; // new float[BatchSize * OutChannel];
    //Err = cudaMemcpy(OutData, d_imgOut, BatchSize * OutChannel * sizeof(float), cudaMemcpyDeviceToHost);
    Err = cudaMemcpy(ImageOutData, d_imgOut, BatchSize * OutChannel * sizeof(float), cudaMemcpyDeviceToHost);
    HANDLE_ERROR(Err);
    Err = cudaMemcpy(ReluOutData, d_reluOut, BatchSize * OutChannel * sizeof(float), cudaMemcpyDeviceToHost);
    HANDLE_ERROR(Err);
    SAFE_DELETE(CBC);
    SAFE_DELETE(CKC);
    Err = cudaFree(d_imgOut);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_reluOut);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_CBC);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_CKC);
    HANDLE_ERROR(Err);
    return OutData;
}

__global__ void SoftMax(SoftMaxKernelCUDA* SMKC, float* InData, float *OutData, int BatchSize, int InChannel, int OutChannel)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < BatchSize && col < OutChannel) {
        float sum = 0;
        for (int i = 0; i < InChannel; i++) {
            sum += InData[row * InChannel + i] * SMKC->W[col * OutChannel + i];  // W[1024*10]，但存储顺序为10*1024
        }
        sum += SMKC->B[col];
        OutData[row * OutChannel + col] = expf(sum);
        SYNC();
        float sum_col = 0;
        for (int i = 0; i < OutChannel; i++) {
            sum_col += OutData[row * OutChannel + i];
        }
        SYNC();
        OutData[row * OutChannel + col] /= sum_col;
        SYNC();
    }
}

extern "C" float* SoftMaxCUDA(SoftMaxKernel* SMK, int mBatchSize, int* mOutChannel, float** mImageOutData)
{
    //test_i(SMK, mBatchSize, *mImageOutData);
    cudaError_t Err;
    SMK->ApplyLastInput(*mImageOutData, mBatchSize * SMK->Row);
    SoftMaxKernelCUDA* SMKC = new SoftMaxKernelCUDA(SMK), * d_SMKC;

    Err = CUDA_MALLOC(d_SMKC, sizeof(SoftMaxKernelCUDA));
    HANDLE_ERROR(Err);
    Err = cudaMemcpy(d_SMKC, SMKC, sizeof(SoftMaxKernelCUDA), cudaMemcpyHostToDevice);
    HANDLE_ERROR(Err);

    float* d_InData, * d_OutData, * OutData;
    int InSize = mBatchSize * SMKC->Row * sizeof(float), OutSize = mBatchSize * SMKC->Column * sizeof(float);
    OutData = new float[OutSize];

    Err = CUDA_MALLOC(d_InData, InSize);
    HANDLE_ERROR(Err);
    Err = CUDA_MALLOC(d_OutData, OutSize);
    HANDLE_ERROR(Err);

    Err = cudaMemcpy(d_InData, *mImageOutData, InSize, cudaMemcpyHostToDevice);
    HANDLE_ERROR(Err);

    dim3 Block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 Grid((SMKC->Column - 1 + BLOCK_SIZE) / BLOCK_SIZE, (mBatchSize - 1 + BLOCK_SIZE) / BLOCK_SIZE);
    SoftMax <<< Grid, Block >>> (d_SMKC, d_InData, d_OutData, mBatchSize, SMKC->Row, SMKC->Column);
    (*mOutChannel) = SMKC->Column;
    Err = cudaMemcpy(OutData, d_OutData, OutSize, cudaMemcpyDeviceToHost);
    HANDLE_ERROR(Err);

    Err = cudaFree(d_InData);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_OutData);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_SMKC);
    HANDLE_ERROR(Err);
    SAFE_DELETE(SMKC);
    (*mImageOutData) = OutData;

    //PRINT(OutData, 10);

    return (*mImageOutData);
}

__global__ void Conv2D(ConvBlocksCUDA* CBC, ConvKernelCUDA* CKC, float* OutData, float* ReluData, int BatchSize, int InChannel, int OutChannel, int ImgSize)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int dep = blockIdx.z * blockDim.z + threadIdx.z;
    int ConvLen = CBC->ArrayLen;
    if (col < ImgSize && row < OutChannel && dep < BatchSize) {

        int DataPos = (dep * ImgSize + col) * ConvLen;
        int ConvPos = row * ConvLen;
        int OutPos = dep * ImgSize * OutChannel + row * ImgSize + col;
        float sum = 0;
        for (int i = 0; i < ConvLen; i++) {
            sum += CBC->ConvArray[DataPos + i] * CKC->W[ConvPos + i];
        }
        sum += CKC->B[row];
        SYNC();
        if (sum > 0.0f) {
            OutData[OutPos] = sum;
            ReluData[OutPos] = 1.0f;
        }

    }
}

extern "C" float* Conv2DCUDA(ConvBlocks * CB, int ConvBlocksPerBatch, ConvKernel * CK, float* ImageOutData, int ImageSize2D, float* ReluOutData, int BatchSize)
{
    //test_CB(CB, CK, BatchSize, nullptr, 0);
    cudaError_t Err;
    int InChannel = CK->ConvKernelChannel, OutChannel = CK->ConvKernelCount;
    int InSize = BatchSize * InChannel * ImageSize2D * sizeof(float), OutSize = BatchSize * OutChannel * ImageSize2D * sizeof(float);
    ConvBlocksCUDA* CBC = new ConvBlocksCUDA(CB, BatchSize, ImageSize2D), *d_CBC;
    ConvKernelCUDA* CKC = new ConvKernelCUDA(CK), *d_CKC;
    Err = CUDA_MALLOC(d_CBC, sizeof(ConvBlocksCUDA));
    HANDLE_ERROR(Err);
    Err = cudaMemcpy(d_CBC, CBC, sizeof(ConvBlocksCUDA), cudaMemcpyHostToDevice);
    HANDLE_ERROR(Err);

    Err = CUDA_MALLOC(d_CKC, sizeof(ConvKernelCUDA));
    HANDLE_ERROR(Err);
    Err = cudaMemcpy(d_CKC, CKC, sizeof(ConvKernelCUDA), cudaMemcpyHostToDevice);
    HANDLE_ERROR(Err);


    float* d_OutData, *d_ReluData;
    float* OutData = nullptr; //new float[OutSize / sizeof(float)];
    Err = CUDA_MALLOC(d_OutData, OutSize);
    HANDLE_ERROR(Err);
    Err = CUDA_MALLOC(d_ReluData, OutSize);
    HANDLE_ERROR(Err);
    dim3 Block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 Grid((ImageSize2D - 1 + BLOCK_SIZE) / BLOCK_SIZE, (OutChannel - 1 + BLOCK_SIZE) / BLOCK_SIZE, BatchSize);
    Conv2D <<< Grid, Block >>> (d_CBC, d_CKC, d_OutData, d_ReluData, BatchSize, InChannel, OutChannel, ImageSize2D);

    //Err = cudaMemcpy(OutData, d_OutData, OutSize, cudaMemcpyDeviceToHost);
    //PRINT(OutData, 100);
    Err = cudaMemcpy(ImageOutData, d_OutData, OutSize, cudaMemcpyDeviceToHost);
    HANDLE_ERROR(Err);
    Err = cudaMemcpy(ReluOutData, d_ReluData, OutSize, cudaMemcpyDeviceToHost);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_OutData);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_ReluData);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_CBC);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_CKC);
    HANDLE_ERROR(Err);
    SAFE_DELETE(CKC);
    SAFE_DELETE(CBC);
    return OutData;
}

__global__ void _MAX_POOL_2_2(ConvKernelCUDA* CKC, int BatchSize, int Depth, int ImgWidth, int ImgHeight, float* InData, float* OutData, float* GradData) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int dep = blockIdx.z * blockDim.z + threadIdx.z;
    if (col < ImgWidth / 2 && row < ImgHeight / 2 && dep < BatchSize * Depth) {
        int OutPos = dep * ImgHeight * ImgWidth / 4 + row * ImgWidth /2 + col;
        float tmp[4];
        int InPos[4];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                int x = col * 2 + j;
                int y = row * 2 + i;
                int pos = dep * ImgHeight * ImgWidth + y * ImgWidth + x;
                tmp[i * 2 + j] = InData[pos];
                InPos[i * 2 + j] = pos;
            }
        }
        SYNC();
        if (tmp[0] == tmp[1] && tmp[1] == tmp[2] && tmp[2] == tmp[3]) {
            OutData[OutPos] = tmp[0];
            for (int i = 0; i < 4; i++) {
                GradData[InPos[i]] = 0.25f;
            }
        }
        else{
            float Max = tmp[0];
            int MaxPos = 0;;
            for (int i = 0; i < 4; i++) {
                if (tmp[i] > Max) {
                    Max = tmp[i];
                    MaxPos = i;
                }
            }
            OutData[OutPos] = Max;
            GradData[InPos[MaxPos]] = 1.0f;
        }
    }
}

extern "C" void Max_Pool_2_2_CUDA(ConvKernel * CK, int BatchSize, int ImgWidth, int ImgHeight, int Depth, float* InData, float* OutData, float* GradData) {

    /*for (int i = 0; i < mBatchSize * CK->ConvKernelCount; i++)
    {
        int InPos = i * OriginalImageSize;
        int OutPos = i * ImageSize;
        Max_Pool_2_2(mInWidth, mInHeight, mImageOutData + InPos, MaxPoolGradientData + InPos, OutData + OutPos);
    }*/

    cudaError_t Err;
    ConvKernelCUDA* CKC = new ConvKernelCUDA(CK);
    float* d_InData, * d_OutData, *d_GradData;
    int Size = BatchSize * Depth * ImgHeight * ImgWidth * sizeof(float);
    Err = CUDA_MALLOC(d_InData, Size);
    HANDLE_ERROR(Err);
    Err = CUDA_MALLOC(d_OutData, Size / 4);
    HANDLE_ERROR(Err);
    Err = CUDA_MALLOC(d_GradData, Size);
    HANDLE_ERROR(Err);
    Err = cudaMemcpy(d_InData, InData, Size, cudaMemcpyHostToDevice);
    HANDLE_ERROR(Err);
    dim3 Block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 Grid((ImgWidth / 2 - 1 + BLOCK_SIZE) / BLOCK_SIZE, (ImgHeight / 2 - 1 + BLOCK_SIZE) / BLOCK_SIZE, (BatchSize * Depth - 1 + BLOCK_SIZE) / BLOCK_SIZE);
    _MAX_POOL_2_2 <<< Grid, Block >>> (CKC, BatchSize, Depth, ImgWidth, ImgHeight, d_InData, d_OutData, d_GradData);
    Err = cudaMemcpy(OutData, d_OutData, Size / 4, cudaMemcpyDeviceToHost);
    HANDLE_ERROR(Err);
    Err = cudaMemcpy(GradData, d_GradData, Size, cudaMemcpyDeviceToHost);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_InData);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_OutData);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_GradData);
    HANDLE_ERROR(Err);
    SAFE_DELETE(CKC);
}


__global__ void _BuildConvArray(ConvBlocksCUDA* d_CBC, float* d_ImageData, int ImageWidth, int ImageHeight,
                            int ImageDepth, int ConvWidth, int ConvHeight, int BatchSize)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int dep = blockIdx.z * blockDim.z + threadIdx.z;


    if (col < ImageWidth && row < ImageHeight && dep < BatchSize) {
        int ConvLen = d_CBC->ArrayLen;
        //int ConvLen = ImageDepth * ConvWidth * ConvHeight;
        int ImgSize2D = ImageWidth * ImageHeight;
        int Batch_Pos = dep * ImgSize2D;
        int HalfConv = ConvWidth >> 1;
        float* ConvQuad = new float[ConvLen];
        for (int k = 0; k < ImageDepth; k++) {
            for (int m = 0; m < ConvHeight; m++) {
                for (int n = 0; n < ConvWidth; n++) {
                    int ConvPos = k * ConvHeight * ConvWidth + m * ConvWidth + n;
                    int x_ = n - HalfConv;
                    int y_ = m - HalfConv;
                    int x = col + x_;
                    int y = row + y_;
                    if (x < 0 || x >= ImageWidth || y < 0 || y >= ImageHeight) {
                        ConvQuad[ConvPos] = 0;
                    }
                    else {
                        int Pos = BatchSize * ImgSize2D * ImageDepth + k * ImgSize2D + x * ImageWidth + y;
                        ConvQuad[ConvPos] = d_ImageData[Pos];
                    }
                }
            }
        }
        SYNC();
        memcpy(d_CBC->ConvArray, ConvQuad, ConvLen * sizeof(float));
        delete[]ConvQuad;
    }
}

extern "C" void BuildConvArrayCUDA(ConvBlocksCUDA *d_CBC, float* d_ImageData, int ImageWidth, int ImageHeight,
                            int ImageDepth, int ConvWidth, int ConvHeight, int BatchSize)
{
    int ConvLen = ConvWidth * ConvHeight * ImageDepth;
    int ImageSize = ImageWidth * ImageHeight * ImageDepth;


    cudaError_t Err;
    dim3 Block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 Grid((ImageWidth - 1 + BLOCK_SIZE) / BLOCK_SIZE, (ImageHeight - 1 + BLOCK_SIZE) / BLOCK_SIZE, BatchSize);


    _BuildConvArray <<< Grid, Block >>> (d_CBC, d_ImageData, ImageWidth, ImageHeight, ImageDepth, ConvWidth, ConvHeight, BatchSize);
    Err = cudaDeviceSynchronize();
    HANDLE_ERROR(Err);
    // 按 batch 划分 convArray 进行计算
    /*for (int i = 0; i < BatchSize; i++) {
        int Pos = i * ImageSize;
        _BuildConvArrayPerBatch(CB, &ImageData[Pos], ImageWidth, ImageHeight, ImageDepth, ConvWidth, ConvHeight, i);
    }*/


    return;
}

extern "C" float* Conv2DCUDA_2(ConvBlocksCUDA *d_CBC, int ConvBlocksPerBatch, ConvKernelCUDA * d_CKC, float* ImageOutData,
                                int ImageSize2D, float* ReluOutData, int BatchSize,
                                int InChannel, int OutChannel)
{
    //test_CB(CB, CK, BatchSize, nullptr, 0);
    cudaError_t Err;

    int InSize = BatchSize * InChannel * ImageSize2D * sizeof(float), OutSize = BatchSize * OutChannel * ImageSize2D * sizeof(float);

    float* OutData = nullptr; //new float[OutSize / sizeof(float)];
    float* d_OutData, * d_ReluData;
    Err = CUDA_MALLOC(d_OutData, OutSize);
    HANDLE_ERROR(Err);
    Err = CUDA_MALLOC(d_ReluData, OutSize);
    HANDLE_ERROR(Err);
    dim3 Block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 Grid((ImageSize2D - 1 + BLOCK_SIZE) / BLOCK_SIZE, (OutChannel - 1 + BLOCK_SIZE) / BLOCK_SIZE, BatchSize);
    Conv2D <<< Grid, Block >>> (d_CBC, d_CKC, d_OutData, d_ReluData, BatchSize, InChannel, OutChannel, ImageSize2D);

    //Err = cudaMemcpy(OutData, d_OutData, OutSize, cudaMemcpyDeviceToHost);
    //PRINT(OutData, 100);
    Err = cudaMemcpy(ImageOutData, d_OutData, OutSize, cudaMemcpyDeviceToHost);
    HANDLE_ERROR(Err);
    Err = cudaMemcpy(ReluOutData, d_ReluData, OutSize, cudaMemcpyDeviceToHost);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_OutData);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_ReluData);
    HANDLE_ERROR(Err);
    return OutData;
}

__global__ void _Conv2D_Both(float* InData, ConvKernelCUDA* CKC, float* OutData, float* ReluData, int BatchSize, int InChannel, int OutChannel, int ImageWidth, int ImageHeight)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int dep = blockIdx.z * blockDim.z + threadIdx.z;

    if (col < ImageWidth && row < ImageHeight && dep < OutChannel * BatchSize) {
        int ChannelIdx = dep % OutChannel, BatchIdx = dep / OutChannel;
        int ImgSize2D = ImageWidth * ImageHeight;
        int ConvWidth = CKC->ConvKernelWidth, ConvHeight = CKC->ConvKernelHeight;
        int ConvLen = CKC->ConvKernelWidth * CKC->ConvKernelHeight * CKC->ConvKernelChannel;
        int HalfConv = ConvWidth >> 1;
        float sum = 0;
        for (int k = 0; k < InChannel; k++) {
            for (int m = 0; m < ConvHeight; m++) {
                for (int n = 0; n < ConvWidth; n++) {
                    int x_ = n - HalfConv;
                    int y_ = m - HalfConv;
                    int x = col + x_;
                    int y = row + y_;
                    if (!(x < 0 || x >= ImageWidth || y < 0 || y >= ImageHeight)) {
                        int ConvPos = k * ConvHeight * ConvWidth + m * ConvWidth + n;
                        int Pos = BatchIdx * ImgSize2D * InChannel + k * ImgSize2D + x * ImageWidth + y;
                        int ConvKernelPos = ChannelIdx * ConvLen + ConvPos;
                        sum += InData[Pos] * CKC->W[ConvKernelPos];
                    }
                }
            }
        }
        sum += CKC->B[ChannelIdx];
        SYNC();
        int OutPos = BatchIdx * ImgSize2D * OutChannel + ChannelIdx * ImgSize2D
                    + row * ImageWidth + col;
        if (sum > 0.0f) {
            OutData[OutPos] = sum;
            ReluData[OutPos] = 1.0f;
        }

    }
}

__global__ void _Conv2D_Opt(float* InData, ConvKernelCUDA* CKC, float* OutData, float* ReluData, int BatchSize, int InChannel, int OutChannel, int ImageWidth, int ImageHeight)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int dep = blockIdx.z * blockDim.z + threadIdx.z;

    if (col < ImageWidth && row < ImageHeight && dep < OutChannel * BatchSize) {
        __shared__ float Data_Share[BLOCK_SIZE * BLOCK_SIZE];
        int ChannelIdx = dep % OutChannel, BatchIdx = dep / OutChannel;
        int ImgSize2D = ImageWidth * ImageHeight;
        int ConvWidth = CKC->ConvKernelWidth, ConvHeight = CKC->ConvKernelHeight;
        int ConvLen = CKC->ConvKernelWidth * CKC->ConvKernelHeight * CKC->ConvKernelChannel;
        int HalfConv = ConvWidth >> 1;
        float sum = 0;
        for (int k = 0; k < InChannel; k++) {
            int InPos = BatchIdx * ImgSize2D * InChannel + k * ImgSize2D + row * ImageWidth + col;
            Data_Share[threadIdx.y * blockDim.x + threadIdx.x] = InData[InPos];
            SYNC();
            int This_tile_start_point_x = blockIdx.x * blockDim.x;
            int Next_tile_start_point_x = (blockIdx.x + 1) * blockDim.x;
            int This_tile_start_point_y = blockIdx.y * blockDim.y;
            int Next_tile_start_point_y = (blockIdx.y + 1) * blockDim.y;
            int N_start_point_x = col - HalfConv;
            int N_start_point_y = row - HalfConv;
            for (int m = 0; m < ConvHeight; m++) {
                for (int n = 0; n < ConvWidth; n++) {
                    int x_index = N_start_point_x + n;
                    int y_index = N_start_point_y + m;
                    int ConvPos = k * ConvHeight * ConvWidth + m * ConvWidth + n;
                    int ConvKernelPos = ChannelIdx * ConvLen + ConvPos;
                    if (x_index >= 0 && x_index < ImageWidth && y_index >= 0 && y_index < ImageHeight) {
                        if (x_index >= This_tile_start_point_x && x_index <= Next_tile_start_point_x &&
                            y_index >= This_tile_start_point_y && y_index <= Next_tile_start_point_y) {
                            sum += Data_Share[(row + m - HalfConv) * ConvWidth + (col + n - HalfConv)] * CKC->W[ConvKernelPos];
                        }
                        else {
                            int Pos = BatchIdx* ImgSize2D* InChannel + k * ImgSize2D + y_index * ImageWidth + x_index;
                            sum += InData[Pos] * CKC->W[ConvKernelPos];
                        }

                    }
                }
            }
        }
        sum += CKC->B[ChannelIdx];
        SYNC();
        int OutPos = BatchIdx * ImgSize2D * OutChannel + ChannelIdx * ImgSize2D
            + row * ImageWidth + col;
        if (sum > 0.0f) {
            OutData[OutPos] = sum;
            ReluData[OutPos] = 1.0f;
        }

    }
}


extern "C" float* Conv2DCUDA_3(float* d_Indata, ConvKernelCUDA * d_CKC, float* ImageOutData,
    int ImageWidth, int ImageHeight, float* ReluOutData, int BatchSize,
    int InChannel, int OutChannel)
{
    //test_CB(CB, CK, BatchSize, nullptr, 0);
    cudaError_t Err;
    int ImageSize2D = ImageWidth * ImageHeight;
    int InSize = BatchSize * InChannel * ImageSize2D * sizeof(float), OutSize = BatchSize * OutChannel * ImageSize2D * sizeof(float);

    float* OutData = nullptr; //new float[OutSize / sizeof(float)];
    float* d_OutData, * d_ReluData;
    Err = CUDA_MALLOC(d_OutData, OutSize);
    HANDLE_ERROR(Err);
    Err = CUDA_MALLOC(d_ReluData, OutSize);
    HANDLE_ERROR(Err);
    dim3 Block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 Grid((ImageSize2D - 1 + BLOCK_SIZE) / BLOCK_SIZE, (OutChannel - 1 + BLOCK_SIZE) / BLOCK_SIZE, BatchSize * OutChannel - 1);
		// 
		_Conv2D_Both <<< Grid, Block >>> (d_Indata, d_CKC, d_OutData, d_ReluData, BatchSize, InChannel, OutChannel, ImageWidth, ImageHeight);
    //_Conv2D_Opt <<< Grid, Block >>> (d_Indata, d_CKC, d_OutData, d_ReluData, BatchSize, InChannel, OutChannel, ImageWidth, ImageHeight);

    //Err = cudaMemcpy(OutData, d_OutData, OutSize, cudaMemcpyDeviceToHost);
    //PRINT(OutData, 100);
    Err = cudaMemcpy(ImageOutData, d_OutData, OutSize, cudaMemcpyDeviceToHost);
    HANDLE_ERROR(Err);
    Err = cudaMemcpy(ReluOutData, d_ReluData, OutSize, cudaMemcpyDeviceToHost);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_OutData);
    HANDLE_ERROR(Err);
    Err = cudaFree(d_ReluData);
    HANDLE_ERROR(Err);

    return OutData;
}

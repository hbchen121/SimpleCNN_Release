/***********************************************
 * File: Common.h
 *
 * Author: CHB
 * Date: 2021-04-06
 *
 * Purpose:
 *
 *
 **********************************************/

#pragma once

#define SAFE_DELETE(p)       { if(p) { delete (p);     (p)=NULL; } }
#define SAFE_DELETE_ARRAY(p) { if(p) { delete[] (p);   (p)=NULL; } }
#define SAFE_RELEASE(p)      { if(p) { (p)->Release(); (p)=NULL; } }

#define			SINGLETON(T)																\
public:																						\
	static T* GetInstance();																\
	static void Release();																	\
protected:																					\
		static T* Instance;

#define			SINGLETON_IMPL(T)															\
T* T::Instance = nullptr;																	\
T* T::GetInstance()																			\
{																							\
	if (Instance == nullptr)																\
	{																						\
		Instance = new T();																	\
	}																						\
	return Instance;																		\
}																							\
void T::Release()																			\
{																							\
	SAFE_DELETE(Instance);																	\
}																					

#define CUDA_MALLOC(dp, size)	cudaMalloc((void**)&dp, size)
#define SYNC	__syncthreads
#define BLOCK_SIZE 16

#define PRINT(arr, len) \
{for(int ii = 0; ii < len; ii++) \
	printf("%.4lf ", arr[ii]);  \
printf("\n");					\
}

#include "cuda_runtime.h"  
#include "device_launch_parameters.h" 
#include <stdio.h>
#include <stdlib.h>

static void HandleError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#define HANDLE_NULL(a) { if(a == null) {  \
                        printf("Host memory failed in %s at line %d\n", \
                            __FILE__, __LINE__); \
                        exit(EXIT_FAILURE);}}

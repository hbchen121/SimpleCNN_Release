/***********************************************
 * File: SdlfCalculator.cpp
 *
 * Author: CHB
 * Date: 2020-04-10
 *
 * Purpose:
 *
 *
 **********************************************/
#include "SdlfCalculator.h"
#include "SdlfCalculatorCPU.h"
#include "SdlfCalculatorCUDA.h"
#include <math.h>



float SdlfCalculator::UCharToFloat(unsigned char C)
{
	// ÏÈ×ªµ½0£¬1
	float f = float(C) / 255.0f;

	f = f * 2.0f - 1.0f;

	return f;
}

SdlfCalculator* SdlfCalculator::CreateCalculator(SdlfCalculatorType SCT)
{
	SdlfCalculator* SC = nullptr;
	if (SCT == SCT_CPU)
	{
		SC = new SdlfCalculatorCPU;
	}
	else if (SCT == SCT_CUDA)
	{
		SC = new SdlfCalculatorCUDA;
	}
	return SC;
}

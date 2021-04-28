/***********************************************
 * File: MnistFileManager.h
 *
 * Author: LYG
 * Date: 2021-04-06
 *
 * Purpose:
 *
 *
 **********************************************/

#pragma once

#include <stdio.h>
#include <xstring>
#include "Common.h"

 /*
	 ������һ�����ݸ�ʽ�������������硣�ĸ��ļ�������ͼƬ�ļ���������ǩ�ļ���
	 ��ǩ�ļ���
	 1��ǰ4���ֽڣ���0��3���ֽڣ���ħ��2049��int�ͣ�0x00000801, ��ˣ�;

	 2��������4���ֽڣ���4��7���ֽڣ��Ǳ�ǩ�ĸ�����60000��10000��

	 3��������ÿ1���ֽ���һ���޷����͵�����ֵΪ0��9��

	 ͼƬ�ļ���
	 1��ǰ4���ֽڣ���0��3���ֽڣ���ħ��2051��int�ͣ�0x00000803, ��ˣ�;

	 2��������4���ֽڣ���4��7���ֽڣ���ͼ��ĸ�����60000��10000����1��ά�ȣ���

	 3��������4���ֽڣ���8��11���ֽڣ���ͼ���ڸ߶����ɶ��ٸ�������ɣ���2��ά�ȣ���28�����أ���

	 4��������4���ֽڣ���12��15���ֽڣ���ͼ���ڿ�����ɶ��ٸ�������ɣ���3��ά�ȣ���28�����أ���

	 5����������һ����ά���飬��ʾ10000����60000���ֱ���28x28�ĻҶ�ͼ��һ�仰��˵����10000x28x28�����أ�ÿ�����ص�ֵΪ0��255��0�Ǳ�����Ϊ��ɫ��255�Ǻ�ɫ����
 */

class MnistFileManager
{
	SINGLETON(MnistFileManager)
public:
	// ���ص��ڴ档����ѵ�������Կ����Ƿֿ��ģ������ϲ��Լ�������Ҫ���û�ȽϺ���
	bool LoadTrainDataToMemory();
	void ReleaseTrainData();
	bool LoadTestDataToMemory();
	void ReleaseTestData();
	// �����ȡ������ѵ��
	void GetImageAndLabelDataInRandom(int BatchCount, unsigned char*& ImageData, unsigned char*& LabelData);
	void GetImageAndLabelDataByIndex(int Index, unsigned char*& ImageData, unsigned char*& LabelData);
	// �����ȡ�о����ܻᵼ��ѵ��������������������ã�һֱ���������λ�ã�Ȼ��ᵼ��loss�����½���Ȼ��������������ط���loss�����ǣ��µ�
	// ���ԣ���������һ��˳��ѵ��
	// ˳���ȡ������ѵ�����߲��ԣ����ﲻ��Ҫ����batch��ԭ�����ڣ���ȡ������һ����ڴ��ָ�룬�����Ҫ��ȡ���پͶ�ȡ����batch��������ʲôӰ�졣����index��ͷ�˲Żᵼ��Խ��
	void GetTestImageAndLabelDataByIndex(int Index, unsigned char*& ImageData, unsigned char*& LabelData);

	int GetTrainDataCount() const;
	int GetTestDataCount() const;
	int GetImageWidth() const;
	int GetImageHeight() const;

	std::string GetApplicationPath();
protected:
	MnistFileManager();
	~MnistFileManager();

	void TransformLabel(unsigned char* OriginalData, unsigned char* OutData, int LabelCount);
private:
	std::string mApplicationPath;

	int mTrainDataCount;
	int mTestDataCount;
	int mImageWidth;
	int mImageHeight;

	unsigned char* mImageTrainData;
	unsigned char* mImageTestData;
	unsigned char* mImageTrainLabelData;
	unsigned char* mImageTestLabelData;
};
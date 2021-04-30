## 简单卷积神经网络的 C++ 实现及 CUDA 加速

### 项目来源
本项目为个人在学习 GPGPU 课程，即《大规模并行处理器实战（第二版）》时的大作业选题，在lygyue的 [SimpleDeepLearningFramework](https://github.com/lygyue/SimpleDeepLearningFramework) 项目上进行了理解和改进，进行了 CUDA C 编写加速，可参考说明 学习报告.pdf。
+ 有理解不了的 CPU 实现部分，可参考 lygue 的[源项目](https://github.com/lygyue/SimpleDeepLearningFramework);
+ CUDA学习可参考[CUDA C 编程学习](https://blog.csdn.net/qq_40491305/category_10861737.html);
+ 《大规模并行处理器实战（第二版）》书籍，[百度云链接](https://pan.baidu.com/s/1ATxcjt2q8qm0tk4RrbCKbg)，提取码:cuda
+ 部分其他参考书籍间 book 文件夹

### 运行环境
+ 环境：Visual Studio 2019
+ 默认使用：Release x64 进行运行
+ 运行前请将Mnist数据集（包含4个文件）存放在对应文件夹中，否则会提醒"Load mnist file failed."
+ 数据集下载地址：http://yann.lecun.com/exdb/mnist/ 或见文件夹 Mnist

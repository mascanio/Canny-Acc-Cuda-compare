NVCC=nvcc
NVCFLAGS=-O3 --use_fast_math -std=c++11 --gpu-architecture=compute_60 --gpu-code=sm_60,compute_60 --compiler-options -march=native,-Ofast

PGCC=pgc++
PGCCFLAGS=-Minfo -fast -acc -ta=nvidia:8.0 -tp=nehalem -cudalibs

LDFLAGS= -lm

EXEC=exec

ACC_FILES=imageProcessACC.cc
C_CC_CUDA_FILES=benchUtils.cc imageProcessCPU.cc imageProcessGPU.cu imageProcessOne.cu imageUtils.cu test.cc main.cu

ACC_OBJS=imageProcessACC.o
C_CC_CUDA_OBJS=benchUtils.o imageProcessCPU.o imageProcessGPU.o imageProcessOne.o imageUtils.o test.o main.o

all: acc cuda main link

acc:
	# ACC COMPILATION
	$(PGCC) $(PGCCFLAGS) -c -o imageProcessACC.o    imageProcessACC.cc

cuda:
	# CUDA AND C++ COMPILATION
	$(NVCC) $(NVCFLAGS)  -c -o benchUtils.o         benchUtils.cc 
	$(NVCC) $(NVCFLAGS)  -c -o imageProcessCPU.o    imageProcessCPU.cc 
	$(NVCC) $(NVCFLAGS)  -c -o imageProcessGPU.o    imageProcessGPU.cu 
	$(NVCC) $(NVCFLAGS)  -c -o imageProcessOne.o    imageProcessOne.cu 
	$(NVCC) $(NVCFLAGS)  -c -o imageUtils.o         imageUtils.cu 
	$(NVCC) $(NVCFLAGS)  -c -o test.o               test.cc

main:
	# MAIN COMPILATION
	$(NVCC) $(NVCFLAGS)  -c -o main.o               main.cu

link: 
	# LINK
	$(PGCC) $(PGCCFLAGS) $(ACC_OBJS) $(C_CC_CUDA_OBJS) -o $(EXEC)

clean:
	rm *.o $(EXEC)
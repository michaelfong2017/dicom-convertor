// dicom-convertor.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include <torch/cuda.h>

#include <vtkDICOMImageReader.h>
#include <vtkImageAppendComponents.h>
#include <vtkImageExtractComponents.h>
#include <vtkImageData.h>
#include <vtkSmartPointer.h>

int main()
{
    std::cout << "Number of CUDA devices available: " << torch::cuda::device_count() << std::endl;
    std::cout << "Whether at least one CUDA device is available: " << torch::cuda::is_available() << std::endl;
    std::cout << "Whether CUDA is available, and CuDNN is available: " << torch::cuda::cudnn_is_available() << std::endl;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

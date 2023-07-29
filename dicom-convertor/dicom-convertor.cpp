// dicom-convertor.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include <vtkDICOMImageReader.h>
#include <vtkImageAppendComponents.h>
#include <vtkImageExtractComponents.h>
#include <vtkImageData.h>
#include <vtkSmartPointer.h>

#include "dcmtk/dcmdata/dcfilefo.h"
#include "dcmtk/dcmdata/dcdatset.h"
#include "dcmtk/dcmdata/dcdeftag.h"

void testCuda() {
    int size = 10000;
    // Create a random tensor on the CPU
    torch::Tensor x_cpu = torch::rand({ size, size });

    // Create a random tensor on the GPU, if available
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Testing on both GPU and CPU." << std::endl;
        device = torch::kCUDA;

        torch::Tensor x_gpu = torch::rand({ size, size }, device);

        // Test the operation on the GPU tensor
        auto start_gpu = std::chrono::high_resolution_clock::now();
        torch::Tensor y_gpu = x_gpu + x_gpu;
        auto end_gpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_gpu = end_gpu - start_gpu;
        std::cout << "Time taken on GPU: " << elapsed_gpu.count() << " seconds" << std::endl;
    }
    else {
        std::cout << "CUDA is unavailable! Testing on CPU only." << std::endl;
    }

    // Test the operation on the CPU tensor
    auto start_cpu = std::chrono::high_resolution_clock::now();
    torch::Tensor y_cpu = x_cpu + x_cpu;
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cpu = end_cpu - start_cpu;
    std::cout << "Time taken on CPU: " << elapsed_cpu.count() << " seconds" << std::endl;
}

int main()
{
    std::cout << "Number of CUDA devices available: " << torch::cuda::device_count() << std::endl;
    std::cout << "Whether at least one CUDA device is available: " << torch::cuda::is_available() << std::endl;
    std::cout << "Whether CUDA is available, and CuDNN is available: " << torch::cuda::cudnn_is_available() << std::endl;

    testCuda();

    DcmFileFormat fileformat;
    if (fileformat.loadFile("C:\\Users\\user\\Documents\\dicom-convertor\\data\\PWHOR190734217S_12Oct2021_CX03WQDU_3DQ.dcm").good() == false) {
        std::cerr << "Error: cannot read DICOM file" << std::endl;
        return 1;
    }

    DcmDataset* dataset = fileformat.getDataset();

    Uint16 columns = 0;
    Uint16 rows = 0;
    if (dataset->findAndGetUint16(DCM_Columns, columns).bad() ||
        dataset->findAndGetUint16(DCM_Rows, rows).bad()) {
        std::cerr << "Error: cannot read image dimensions from DICOM file\n";
        return 1;
    }

    Uint32 slices;
    if (dataset->findAndGetUint32(DcmTagKey(0x3001, 0x1001), slices).bad()) {
        std::cerr << "Error: cannot read number of slices from DICOM file\n";
        return 1;
    }

    double physicalDeltaX = 0;
    double physicalDeltaY = 0;
    double physicalDeltaZ = 0;
    if (dataset->findAndGetFloat64(DCM_PhysicalDeltaX, physicalDeltaX).bad() ||
        dataset->findAndGetFloat64(DCM_PhysicalDeltaY, physicalDeltaY).bad() ||
        dataset->findAndGetFloat64(DcmTagKey(0x3001, 0x1003), physicalDeltaZ).bad()) {
        std::cerr << "Error: cannot read pixel spacing from DICOM file\n";
        return 1;
    }

    double spacing[3] = { physicalDeltaX * 10, physicalDeltaY * 10, physicalDeltaZ * 10 };

    Sint32 numberOfFrames = 0;
    if (dataset->findAndGetSint32(DcmTagKey(0x0028, 0x0008), numberOfFrames).bad()) {
        std::cerr << "Error: cannot read number of frames from DICOM file\n";
        return 1;
    }

    double frameTimeMsec = 0;
    if (dataset->findAndGetFloat64(DcmTagKey(0x0018, 0x1063), frameTimeMsec).bad()) {
        std::cerr << "Error: cannot read frame time from DICOM file\n";
        return 1;
    }

    std::vector<int> pixelShape{static_cast<int>(numberOfFrames), static_cast<int>(slices),
        static_cast<int>(rows), static_cast<int>(columns)};
    std::cout << "Pixel shape: (" << pixelShape[0] << ", " << pixelShape[1] << ", " << pixelShape[2] << ", "
        << pixelShape[3] << ")\n";

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

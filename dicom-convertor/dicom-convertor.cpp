// dicom-convertor.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>
#include <vector>

#if __has_include(<filesystem>)
#include <filesystem>
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
#endif

#include <torch/torch.h>

#include <vtkimagereader.h>
#include <vtksmartpointer.h>
#include <vtkimagedata.h>
#include <vtkpointdata.h>
#include <vtkdataarray.h>
#include <vtkunsignedchararray.h>

#include "dcmtk/dcmdata/dcfilefo.h"
#include "dcmtk/dcmdata/dcdatset.h"
#include "dcmtk/dcmdata/dcdeftag.h"

#include <Eigen/Dense>

using namespace std;

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

std::vector<std::vector<std::vector<unsigned char>>> rotate3DVector(const std::vector<std::vector<std::vector<unsigned char>>>& imageArray) {
    size_t rows = imageArray.size();
    size_t columns = imageArray[0].size();
    size_t slices = imageArray[0][0].size();

    // Create a rotatedArray with the dimensions swapped
    std::vector<std::vector<std::vector<unsigned char>>> rotatedArray(slices, std::vector<std::vector<unsigned char>>(columns, std::vector<unsigned char>(rows)));

    // Perform rotation
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < columns; ++j) {
            for (size_t k = 0; k < slices; ++k) {
                rotatedArray[k][j][rows - 1 - i] = imageArray[i][j][k];
            }
        }
    }

    return rotatedArray;
}

std::vector<std::vector<std::vector<unsigned char>>> flip3DVector(const std::vector<std::vector<std::vector<unsigned char>>>& array) {
    std::vector<std::vector<std::vector<unsigned char>>> flippedArray;
    for (const auto& slice : array) {
        std::vector<std::vector<unsigned char>> flippedSlice(slice.rbegin(), slice.rend());
        flippedArray.push_back(flippedSlice);
    }
    return flippedArray;
}

void print3DVector(const std::vector<std::vector<std::vector<unsigned char>>>& imageArray) {
    for (const auto& slice : imageArray) {
        for (const auto& row : slice) {
            for (const auto& element : row) {
                std::cout << static_cast<int>(element) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void testRot90AndFlip() {
    // Sample 3D imageArray
    std::vector<std::vector<std::vector<unsigned char>>> imageArray = {
        {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
        {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}
    };
    std::cout << "Shape of original vector: (" << imageArray.size() << ", " << imageArray[0].size() << ", " << imageArray[0][0].size() << ")\n";
    print3DVector(imageArray);

    /* Expected output
    Shape of original vector: (2, 3, 4)
    1 2 3 4
    5 6 7 8
    9 10 11 12

    13 14 15 16
    17 18 19 20
    21 22 23 24
    */

    // Rotate the imageArray
    imageArray = rotate3DVector(imageArray);
    std::cout << "Shape of rotated vector: (" << imageArray.size() << ", " << imageArray[0].size() << ", " << imageArray[0][0].size() << ")\n";
    print3DVector(imageArray);

    /* Expected output
    Shape of rotated vector: (4, 3, 2)
    13 1
    17 5
    21 9

    14 2
    18 6
    22 10

    15 3
    19 7
    23 11

    16 4
    20 8
    24 12
    */

    // Flip the imageArray along the second axis
    imageArray = flip3DVector(imageArray);
    std::cout << "Shape of flipped vector: (" << imageArray.size() << ", " << imageArray[0].size() << ", " << imageArray[0][0].size() << ")\n";
    print3DVector(imageArray);

    /* Expected output
    Shape of flipped vector: (4, 3, 2)
    21 9
    17 5
    13 1

    22 10
    18 6
    14 2

    23 11
    19 7
    15 3

    24 12
    20 8
    16 4
    */
}

int main()
{
    std::cout << "Current path is " << std::filesystem::current_path() << std::endl; // (1)

    std::cout << "Number of CUDA devices available: " << torch::cuda::device_count() << std::endl;
    std::cout << "Whether at least one CUDA device is available: " << torch::cuda::is_available() << std::endl;
    std::cout << "Whether CUDA is available, and CuDNN is available: " << torch::cuda::cudnn_is_available() << std::endl;

    testCuda();

    testRot90AndFlip();
    //

    auto t1 = std::chrono::high_resolution_clock::now();

    std::string filePath = "data\\PWHOR190734217S_12Oct2021_CX03WQDU_3DQ.dcm";

    DcmFileFormat fileformat;
    if (fileformat.loadFile(filePath.c_str()).good() == false) {
        std::cerr << "Error: cannot read DICOM file" << std::endl;
        return 1;
    }

    DcmDataset* dataset = fileformat.getDataset();

    Uint16 columnsRaw = 0;
    Uint16 rowsRaw = 0;
    if (dataset->findAndGetUint16(DCM_Columns, columnsRaw).bad() ||
        dataset->findAndGetUint16(DCM_Rows, rowsRaw).bad()) {
        std::cerr << "Error: cannot read image dimensions from DICOM file\n";
        return 1;
    }

    Uint32 slicesRaw;
    if (dataset->findAndGetUint32(DcmTagKey(0x3001, 0x1001), slicesRaw).bad()) {
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

    Sint32 framesRaw = 0;
    if (dataset->findAndGetSint32(DcmTagKey(0x0028, 0x0008), framesRaw).bad()) {
        std::cerr << "Error: cannot read number of frames from DICOM file\n";
        return 1;
    }

    double frameTimeMsec = 0;
    if (dataset->findAndGetFloat64(DcmTagKey(0x0018, 0x1063), frameTimeMsec).bad()) {
        std::cerr << "Error: cannot read frame time from DICOM file\n";
        return 1;
    }

    int frames = static_cast<int>(framesRaw);
    int slices = static_cast<int>(slicesRaw);
    int rows = static_cast<int>(rowsRaw);
    int columns = static_cast<int>(columnsRaw);
    std::cout << "Pixel shape: (" << frames << ", " << slices << ", " << rows << ", " << columns << ")\n";

    // After pixel shape
    //
    //
    int pixelSize = frames * slices * rows * columns;
    std::filesystem::path p{filePath};
    int totalFileSize = std::filesystem::file_size(p);
    int headerSize = totalFileSize - pixelSize;

    std::cout << "The total file size is " << totalFileSize << " bytes." << std::endl;
    std::cout << "The header size is " << headerSize << " bytes." << std::endl;
    std::cout << "The pixel size is " << pixelSize << " bytes." << std::endl;

    
    std::vector<std::vector<std::vector<std::vector<unsigned char>>>> image4D(frames,
        std::vector<std::vector<std::vector<unsigned char>>>(columns,
            std::vector<std::vector<unsigned char>>(rows,
                std::vector<unsigned char>(slices, 0))));

    for (int frame = 0; frame < frames; ++frame) {
        vtkSmartPointer<vtkImageReader> imgReader = vtkSmartPointer<vtkImageReader>::New();
        imgReader->SetFileDimensionality(3);
        imgReader->SetFileName(filePath.c_str());
        imgReader->SetNumberOfScalarComponents(1);
        imgReader->SetDataScalarTypeToUnsignedChar();
        imgReader->SetDataExtent(0, columns - 1, 0, rows - 1, 0, slices - 1);
        imgReader->SetHeaderSize(headerSize + frame * slices * rows * columns);
        imgReader->FileLowerLeftOn();
        imgReader->Update();

        double timeStampSec = frame * frameTimeMsec * 0.001;

        vtkSmartPointer<vtkImageData> img = imgReader->GetOutput();
        vtkDataArray* scalars = img->GetPointData()->GetScalars();
        vtkUnsignedCharArray* ucharScalars = vtkUnsignedCharArray::SafeDownCast(scalars);

        std::vector<std::vector<std::vector<unsigned char>>> imageArray(slices, std::vector<std::vector<unsigned char>>(rows, std::vector<unsigned char>(columns)));

        vtkIdType scalarIndex = 0;
        for (int slice = 0; slice < slices; ++slice) {
            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < columns; ++col) {
                    imageArray[slice][row][col] = ucharScalars->GetValue(scalarIndex);
                    scalarIndex++;
                }
            }
        }

        imageArray = rotate3DVector(imageArray);
        imageArray = flip3DVector(imageArray);
        image4D[frame] = imageArray;
    }

    std::cout << "Shape of image4D: " << image4D.size() << " x " << image4D[0].size() << " x " << image4D[0][0].size() << " x " << image4D[0][0][0].size() << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsedTime = t2 - t1;
    std::cout << "Time for dicom to array: " << elapsedTime.count() << std::endl;
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

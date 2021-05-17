//
//  mnist-dataset.swift
//  MetalNNDemo
//
//  Created by Tao Zhang on 2021/5/11.
//

import Foundation
import MetalPerformanceShaders

enum MnistDataSetError : Error {
    case readFileFailed
    case badFileFormat
    case badFileSize
    case badImageCount
}

private func decompressGzipFile(_ fileName: String) -> (UnsafeMutablePointer<UInt8>?, Int) {
    let outputBuffer = UnsafeMutablePointer<UnsafeMutablePointer<UInt8>?>.allocate(capacity: 1)
    outputBuffer.pointee = nil
    
    NSLog("Decompressing %@", fileName)
    
    let fullPath = Bundle.main.resourcePath! + "/data/" + fileName
    let outputSize = fullPath.withCString({(c_fileName) -> Int in
        return C_decompressGzipFile(c_fileName, outputBuffer);
    })
    
    NSLog("%@ decompressed as %d bytes.", fileName, outputSize)
    
    return (outputBuffer.pointee, outputSize)
}

class ImageSet {
    private var imageSetDescriptor = ImageSetDescriptor()
    private var rawImageData : UnsafeMutablePointer<UInt8>? = nil
    private var imageDataPointer : UnsafeMutablePointer<UInt8>? = nil
    
    var count : Int {
        return imageSetDescriptor.count
    }
    
    var width : Int {
        return imageSetDescriptor.width
    }
    
    var height : Int {
        return imageSetDescriptor.height
    }
    
    var stride : Int {
        return imageSetDescriptor.stride
    }
    
    var labels : [Int] = []
    
    deinit {
        rawImageData?.deallocate()
    }
    
    func loadData(imageFile: String, labelFile: String) throws {
        // Read files
        let (imageData, imageDataSize) = decompressGzipFile(imageFile)
        
        if (imageData == nil || imageDataSize <= 0) {
            NSLog("Failed to load file %@", imageFile)
            throw MnistDataSetError.readFileFailed
        }
        
        let (labelData, labelDataSize) = decompressGzipFile(labelFile)
        
        if (labelData == nil || labelDataSize <= 0) {
            imageData?.deallocate()
            NSLog("Failed to load file %@", labelFile)
            throw MnistDataSetError.readFileFailed
        }
        
        // Read descriptors
        if (C_readImageSetDescriptor(imageData!, imageDataSize, &self.imageSetDescriptor) == 0) {
            imageData?.deallocate()
            labelData?.deallocate()
            NSLog("Bad image file format: %@", imageFile)
            throw MnistDataSetError.badFileFormat
        }
        
        var labelSetDescriptor = LabelSetDescriptor()
        
        if (C_readLabelSetDescriptor(labelData!, labelDataSize, &labelSetDescriptor) == 0) {
            imageData?.deallocate()
            labelData?.deallocate()
            NSLog("Bad label file format: %@", labelFile)
            throw MnistDataSetError.badFileFormat
        }
        
        // Free up memory
        if (self.rawImageData != nil) {
            self.rawImageData!.deallocate()
            self.rawImageData = nil
            self.imageDataPointer = nil
            self.labels = []
        }
        
        // Check if the data count matches
        if (self.imageSetDescriptor.count != labelSetDescriptor.count) {
            imageData?.deallocate()
            labelData?.deallocate()
            self.imageSetDescriptor = ImageSetDescriptor()
            NSLog("The item count in the image set mismatches which in the label set.")
            throw MnistDataSetError.badImageCount
        }
        
        // Check file sizes
        if (self.imageSetDescriptor.offset + self.imageSetDescriptor.stride * self.imageSetDescriptor.count !=
                imageDataSize) {
            imageData?.deallocate()
            labelData?.deallocate()
            self.imageSetDescriptor = ImageSetDescriptor()
            NSLog("Bad image file size: %@", imageFile)
            throw MnistDataSetError.badFileSize
        }
        
        if (labelSetDescriptor.offset + labelSetDescriptor.stride * labelSetDescriptor.count !=
                labelDataSize) {
            imageData?.deallocate()
            labelData?.deallocate()
            self.imageSetDescriptor = ImageSetDescriptor()
            NSLog("Bad label file size: %@", labelFile)
            throw MnistDataSetError.badFileSize
        }
        
        // Save the raw image data pointer into the class member
        self.rawImageData = imageData
        
        // Set the image data pointer
        self.imageDataPointer = self.rawImageData!.advanced(by: self.imageSetDescriptor.offset)
        
        // Store labels
        for i in 0 ..< self.imageSetDescriptor.count {
            self.labels.append((Int)(labelData!.advanced(
                                        by: labelSetDescriptor.offset
                                            + i * labelSetDescriptor.stride).pointee))
        }
        
        // Free up the label data buffer
        labelData?.deallocate()
    }
    
    func getRandomTrainingBatchWithDevice(device: MTLDevice,
                                          batchSize: Int,
                                          lossLabelsBatch: inout [MPSCNNLossLabels]) -> [MPSImage] {
        let imageDescriptor = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.unorm8,
                                                 width: self.width,
                                                 height: self.height,
                                                 featureChannels: 1,
                                                 numberOfImages: 1,
                                                 usage: [.shaderWrite,
                                                         .shaderRead])
        var trainBatch : [MPSImage] = []
        lossLabelsBatch.removeAll()
        
        // Allocate memory for labels
        let labelsArrayLength = 12  // the closest multiple of 4 greater than 10.
        let labelsRawData = UnsafeMutablePointer<Float32>.allocate(capacity: labelsArrayLength)
        let labelsData = Data(bytesNoCopy: labelsRawData,
                              count: labelsArrayLength * MemoryLayout<Float32>.size,
                              deallocator: .free)
        
        // Fill data
        for _ in 0 ..< batchSize {
            let index = Int.random(in: 0 ..< self.count)
            let trainImage = MPSImage(device: device,
                                      imageDescriptor: imageDescriptor)
            trainImage.label = "train-image-" + String(format: "%d", index)
            
            trainImage.writeBytes(
                imageDataPointer!.advanced(by: index * stride),
                dataLayout: .HeightxWidthxFeatureChannels
                , imageIndex: 0)
            
            trainBatch.append(trainImage)
            
            for i in 0 ..< labelsArrayLength {
                var oneHotValue : Float32 = 0.0
                if i == labels[index] {
                    oneHotValue = 1.0
                }
                labelsRawData.advanced(by: i).initialize(to: oneHotValue)
            }
            
            let labelsDescriptor = MPSCNNLossDataDescriptor(data: labelsData,
                                                            layout: .HeightxWidthxFeatureChannels,
                                                            size: MTLSizeMake(1, 1, labelsArrayLength));
            
            let lossLabels = MPSCNNLossLabels(device: device,
                                              labelsDescriptor: labelsDescriptor!)
            
            lossLabelsBatch.append(lossLabels)
        }
        
        return trainBatch
    }
}

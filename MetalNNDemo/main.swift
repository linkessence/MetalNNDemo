//
//  main.swift
//  MetalNNDemo
//
//  Created by Tao Zhang on 2021/5/11.
//

import Foundation
import MetalPerformanceShaders

// Data sets
let TRAIN_IMAGES_SET = "train-images-idx3-ubyte.gz"
let TRAIN_LABELS_SET = "train-labels-idx1-ubyte.gz"
let TEST_IMAGES_SET = "t10k-images-idx3-ubyte.gz"
let TEST_LABELS_SET = "t10k-labels-idx1-ubyte.gz"

// Configuration
let BATCH_SIZE                      : Int     = 40
let TRAINING_ITARATIONS             : Int     = 300

// Initialize the logger
initLogger()

let trainSet = ImageSet()
let testSet = ImageSet()

do {
    try trainSet.loadData(imageFile: TRAIN_IMAGES_SET, labelFile: TRAIN_LABELS_SET)
    try testSet.loadData(imageFile: TEST_IMAGES_SET, labelFile: TEST_LABELS_SET)
} catch {
    NSLog("Failed to load data set.")
    exit(1)
}

// Datasets loaded
NSLog("Datasets loaded.")
NSLog("Training set: %d images of (%d x %d x 1).",
      trainSet.count, trainSet.width, trainSet.height)
NSLog("Test set:     %d images of (%d x %d x 1).",
      testSet.count, testSet.width, testSet.height)

// Get the device and command queue
let device = MTLCreateSystemDefaultDevice()
if (device == nil) {
    NSLog("Failed to create metal device.")
    exit(2)
}

NSLog("%@", device!.description)

let commandQueue = device!.makeCommandQueue()

if (commandQueue == nil) {
    NSLog("Failed to make command queue")
    exit(2)
}

// Classifier model
let classifierGraph = MNISTClassifierGraph(device: device!,
                                           commandQueue: commandQueue!)

func lossReduceSumAcrossBatch(_ imageBatch: [MPSImage]) -> Float32 {
    var ret : Float32 = 0.0;
    for i in 0 ..< imageBatch.count {
        let curr = imageBatch[i];
        assert(curr.width * curr.height * curr.featureChannels == 1)
        
        let pointer = UnsafeMutablePointer<Float32>.allocate(capacity: MemoryLayout<Float32>.size)
        
        curr.readBytes(pointer,
                       dataLayout: .HeightxWidthxFeatureChannels,
                       imageIndex: 0)
        
        ret += pointer.pointee / Float32(BATCH_SIZE);
        pointer.deallocate()
    }
    
    return ret;
}

// Training

var lastCommandBuffer : MPSCommandBuffer? = nil
let doubleBufferingSemaphore = DispatchSemaphore(value: 1)

for i in 0 ..< TRAINING_ITARATIONS {
    doubleBufferingSemaphore.wait()
    var traingLossLabels : [MPSCNNLossLabels] = []
    
    let trainingImageBatch = trainSet.getRandomTrainingBatchWithDevice(device: device!,
                                                                       batchSize: BATCH_SIZE,
                                                                       lossLabelsBatch: &traingLossLabels)
    let commandBuffer = MPSCommandBuffer(from: commandQueue!)
    
    let returnBatch = classifierGraph.encodeTrainingBatchToCommandBuffer(commandBuffer: commandBuffer,
                                                                         sourceImages: trainingImageBatch,
                                                                         lossStates: traingLossLabels)
    
    var outputBatch : [MPSImage] = []
    
    for j in 0 ..< BATCH_SIZE {
        outputBatch.append(traingLossLabels[j].lossImage())
    }
    
    commandBuffer.addCompletedHandler({commandBuffer in
        let traingLoss = lossReduceSumAcrossBatch(outputBatch)
        NSLog("Training batch %d: loss = %f", i, traingLoss)
        if (commandBuffer.error != nil) {
            NSLog("Command buffer error: %@", commandBuffer.error!.localizedDescription)
        }
        doubleBufferingSemaphore.signal()
    })
    
    // Transfer data from GPU to CPU (will be a no-op on embedded GPUs).
    MPSImageBatchSynchronize(returnBatch, commandBuffer);
    // MPSImageBatchSynchronize(outputBatch, commandBuffer);
    
    commandBuffer.commit()
    lastCommandBuffer = commandBuffer
    
}

lastCommandBuffer?.waitUntilCompleted()

NSLog("Done.")



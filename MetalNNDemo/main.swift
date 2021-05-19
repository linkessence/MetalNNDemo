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
let TRAINING_ITARATIONS             : Int     = 1000
let TEST_SET_EVAL_INTERVAL          : Int     = 100

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
    NSLog("Failed to make command queue.")
    exit(2)
}

// Classifier model
let classifierGraph = MNISTClassifierGraph(device: device!,
                                           commandQueue: commandQueue!)

func lossReduceSumAcrossBatch(_ imageBatch: [MPSImage]) -> Float32 {
    var ret : Float32 = 0.0
    for i in 0 ..< imageBatch.count {
        let curr = imageBatch[i]
        assert(curr.width * curr.height * curr.featureChannels == 1)
        
        let pointer = UnsafeMutablePointer<Float32>.allocate(capacity: MemoryLayout<Float32>.size)
        
        curr.readBytes(pointer,
                       dataLayout: .HeightxWidthxFeatureChannels,
                       imageIndex: 0)
        
        defer {
            pointer.deallocate()
        }
        ret += pointer.pointee / Float32(BATCH_SIZE);
    }
    
    return ret
}

func evaluateDigitLabels(labelBatch: [MPSImage],
                         groundTruth: [Int],
                         correctCount: inout Int,
                         totalCount: inout Int) -> Float32 {
    for i in 0 ..< labelBatch.count {
        let label = labelBatch[i]
        let pointer = UnsafeMutablePointer<Float32>.allocate(capacity: MemoryLayout<Float32>.size *
                                                                label.featureChannels)
        defer {
            pointer.deallocate()
        }
        
        label.readBytes(pointer,
                        dataLayout: .HeightxWidthxFeatureChannels,
                        imageIndex: 0)
        
        var maxValue : Float32 = -1000.0
        var argMaxLabel : Int = -1
        
        for j in 0 ..< label.featureChannels {
            let value = pointer.advanced(by: j).pointee
            if value > maxValue {
                maxValue = value
                argMaxLabel = j
            }
        }
        
        if argMaxLabel == groundTruth[i] {
            correctCount += 1
        }
    }
    
    totalCount += labelBatch.count
    return Float(correctCount) / Float(totalCount)
}

// Evaluate the test set.
func evaluateTestSet() {
    // Reload data for the classifier graph
    classifierGraph.inferenceGraph!.reloadFromDataSources()
    
    let numBatches = testSet.count / BATCH_SIZE
    let semaphore = DispatchSemaphore(value: 1)
    
    var lastCommandBuffer : MPSCommandBuffer? = nil
    
    var correctCount : Int = 0
    var totalCount : Int = 0
    
    for i in 0 ..< numBatches {
        semaphore.wait()
        let imageBatch = testSet.getTestBatchWithDevice(device: device!,
                                                        startIndex: i * BATCH_SIZE,
                                                        batchSize: BATCH_SIZE)
        
        let commandBuffer = MPSCommandBuffer(from: commandQueue!)
        
        let outputBatch = classifierGraph.encodeInferenceBatchToCommandBuffer(commandBuffer: commandBuffer,
                                                                              sourceImages: imageBatch)
        
        commandBuffer.addCompletedHandler({commandBuffer in
            assert (outputBatch.count == BATCH_SIZE)
            
            if (commandBuffer.error != nil) {
                NSLog("Command buffer error: %@", commandBuffer.error!.localizedDescription)
            } else {
                let groundTruth = testSet.getTestLabelSlice(startIndex: i * BATCH_SIZE,
                                                            batchSize: BATCH_SIZE)
                
                let _ = evaluateDigitLabels(labelBatch: outputBatch,
                                            groundTruth: groundTruth,
                                            correctCount: &correctCount,
                                            totalCount: &totalCount)
            }
            semaphore.signal()
        })
        
        commandBuffer.commit()
        lastCommandBuffer = commandBuffer
    }
    
    lastCommandBuffer?.waitUntilCompleted()
    NSLog("Test set correctness: %f", Float32(correctCount) / Float32(totalCount))
}

// Training.

var lastCommandBuffer : MPSCommandBuffer? = nil
let doubleBufferingSemaphore = DispatchSemaphore(value: 1)

for i in 0 ..< TRAINING_ITARATIONS {
    doubleBufferingSemaphore.wait()
    var traingLossLabels : [MPSCNNLossLabels] = []
    
    let trainingImageBatch = trainSet.getRandomTrainingBatchWithDevice(device: device!,
                                                                       batchSize: BATCH_SIZE,
                                                                       lossLabelsBatch: &traingLossLabels)
    let commandBuffer = MPSCommandBuffer(from: commandQueue!)
    
    let _ = classifierGraph.encodeTrainingBatchToCommandBuffer(commandBuffer: commandBuffer,
                                                               sourceImages: trainingImageBatch,
                                                               lossStates: traingLossLabels)
    
    var outputBatch : [MPSImage] = []
    
    for j in 0 ..< BATCH_SIZE {
        outputBatch.append(traingLossLabels[j].lossImage())
    }
    
    // Transfer data from the GPU to the CPU (will be a no-op on embedded GPUs)
    #if os(macOS)
    MPSImageBatchSynchronize(outputBatch, commandBuffer)
    #endif
    
    commandBuffer.addCompletedHandler({commandBuffer in
        let traingLoss = lossReduceSumAcrossBatch(outputBatch)
        NSLog("Training batch %d: loss = %f", i, traingLoss)
        if (commandBuffer.error != nil) {
            NSLog("Command buffer error: %@", commandBuffer.error!.localizedDescription)
        }
        doubleBufferingSemaphore.signal()
    })
    
    commandBuffer.commit()
    
    if (i + 1) % TEST_SET_EVAL_INTERVAL == 0 {
        commandBuffer.waitUntilCompleted()
        NSLog("Evaluating test set at iteration %d.", i)
        evaluateTestSet()
    }
    
    lastCommandBuffer = commandBuffer
}

lastCommandBuffer?.waitUntilCompleted()

NSLog("Done.")



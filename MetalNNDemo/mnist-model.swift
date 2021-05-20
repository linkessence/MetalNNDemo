//
//  mnist-model.swift
//  MetalNNDemo
//
//  Created by Tao Zhang on 2021/5/13.
//

import Foundation
import MetalPerformanceShaders

class MNISTClassifierGraph : NSObject {
    let device : MTLDevice
    let commandQueue : MTLCommandQueue
    
    private let _conv1Weights : ConvDataSource
    private let _conv2Weights : ConvDataSource
    private let _fc1Weights : ConvDataSource
    private let _fc2Weights : ConvDataSource
    
    var trainingGraph : MPSNNGraph?
    var inferenceGraph : MPSNNGraph?
    
    init(device : MTLDevice,
         commandQueue : MTLCommandQueue) {
        self.device = device
        self.commandQueue = commandQueue
        
        _conv1Weights = ConvDataSource(kernelWidth: 5,
                                       kernelHeight: 5,
                                       inputFeatureChannels: 1,
                                       outputFeatureChannels: 32,
                                       stride: 1,
                                       label: "conv1",
                                       device: device,
                                       commandQueue: commandQueue)
        
        _conv2Weights = ConvDataSource(kernelWidth: 5,
                                       kernelHeight: 5,
                                       inputFeatureChannels: 32,
                                       outputFeatureChannels: 64,
                                       stride: 1,
                                       label: "conv2",
                                       device: device,
                                       commandQueue: commandQueue)
        
        _fc1Weights = ConvDataSource(kernelWidth: 7,
                                     kernelHeight: 7,
                                     inputFeatureChannels: 64,
                                     outputFeatureChannels: 1024,
                                     stride: 1,
                                     label: "fc1",
                                     device: device,
                                     commandQueue: commandQueue)
        
        _fc2Weights = ConvDataSource(kernelWidth: 1,
                                     kernelHeight: 1,
                                     inputFeatureChannels: 1024,
                                     outputFeatureChannels: 10,
                                     stride: 1,
                                     label: "fc2",
                                     device: device,
                                     commandQueue: commandQueue)
        
        super.init()
        
        initializeTrainingGraph()
        initializeInferenceGraph()
    }
    
    private func initializeTrainingGraph() {
        let finalNode = createNodes(isTraining: true)
        let lossExitPoints = finalNode.trainingGraph(withSourceGradient: nil,
                                                     nodeHandler: {
                                                        (gradientNode,
                                                         inferenceNode,
                                                         inferenceSource,
                                                         gradientSource) in
                                                        
                                                        // Setting the gradient node precision to fp32 since small
                                                        // gradient values could flush to 0 with the limited range of
                                                        // fp16.
                                                        gradientNode.resultImage.format = .float32
                                                     })
        assert(lossExitPoints != nil && lossExitPoints!.count == 1)
        
        trainingGraph = MPSNNGraph(device: self.device,
                                   resultImage: lossExitPoints![0].resultImage,
                                   resultImageIsNeeded: true)
        trainingGraph!.format = .float32
    }
        
    
    private func initializeInferenceGraph() {
        let finalNode = createNodes(isTraining: false)
        inferenceGraph = MPSNNGraph(device: self.device,
                                    resultImage: finalNode.resultImage,
                                    resultImageIsNeeded: true)
        inferenceGraph!.format = .float32
    }
    
    private func createNodes(isTraining: Bool) -> MPSNNFilterNode {
        let sameConvPadding = MPSNNDefaultPadding(method: [.sizeSame,
                                                           .centered])
        
        let samePoolingPadding = MPSNNDefaultPadding.forTensorflowAveragePooling()
        
        let conv1Node = MPSCNNConvolutionNode(source: MPSNNImageNode(handle: nil),
                                              weights: _conv1Weights)
        conv1Node.paddingPolicy = sameConvPadding
        let relu1Node = MPSCNNNeuronReLUNode(source: conv1Node.resultImage)
        
        let pool1Node = MPSCNNPoolingMaxNode(source: relu1Node.resultImage,
                                             filterSize: 2,
                                             stride: 2)
        pool1Node.paddingPolicy = samePoolingPadding
        
        let conv2Node = MPSCNNConvolutionNode(source: pool1Node.resultImage,
                                              weights: _conv2Weights)
        conv2Node.paddingPolicy = sameConvPadding
        
        let relu2Node = MPSCNNNeuronReLUNode(source: conv2Node.resultImage)
        
        let pool2Node = MPSCNNPoolingMaxNode(source: relu2Node.resultImage,
                                             filterSize: 2,
                                             stride: 2)
        pool2Node.paddingPolicy = samePoolingPadding
        
        let fc1Node = MPSCNNFullyConnectedNode(source: pool2Node.resultImage,
                                               weights: _fc1Weights)
        
        let relu3Node = MPSCNNNeuronReLUNode(source: fc1Node.resultImage)
        
        var fc2InputNode : MPSNNFilterNode = relu3Node;
        
        if (isTraining) {
            let dropoutNode = MPSCNNDropoutNode(source: relu3Node.resultImage,
                                                keepProbability: 0.5,
                                                seed: 1,
                                                maskStrideInPixels: MTLSize(width: 1, height: 1, depth: 1))
            fc2InputNode = dropoutNode
        }
        
        let fc2Node = MPSCNNFullyConnectedNode(source: fc2InputNode.resultImage,
                                               weights: _fc2Weights)
        if (isTraining) {
            let lossDescriptor = MPSCNNLossDescriptor(type: MPSCNNLossType.softMaxCrossEntropy,
                                                      reductionType: MPSCNNReductionType.sum)
            lossDescriptor.weight = 1.0 / Float32(BATCH_SIZE)
            
            let lossNode = MPSCNNLossNode(source: fc2Node.resultImage,
                                          lossDescriptor: lossDescriptor)
            
            return lossNode
        } else {
            let softMaxNode = MPSCNNSoftMaxNode(source: fc2Node.resultImage)
            return softMaxNode
        }
    }
    
    func encodeTrainingBatchToCommandBuffer(commandBuffer: MPSCommandBuffer,
                                            sourceImages: [MPSImage],
                                            lossStates: [MPSCNNLossLabels]) -> [MPSImage] {
        
        let returnImages = trainingGraph!.encodeBatch(to: commandBuffer,
                                                      sourceImages: [sourceImages],
                                                      sourceStates: [lossStates])!
        
        MPSImageBatchSynchronize(returnImages, commandBuffer)
        
        return returnImages
    }
    
    func encodeInferenceBatchToCommandBuffer(commandBuffer: MPSCommandBuffer,
                                             sourceImages: [MPSImage]) -> [MPSImage] {
        let outputImages = inferenceGraph!.encodeBatch(to: commandBuffer,
                                                       sourceImages: [sourceImages],
                                                       sourceStates: nil)
        
        assert (outputImages != nil && outputImages?.count == sourceImages.count)
        MPSImageBatchSynchronize(outputImages!, commandBuffer)
        
        return outputImages!
    }
}

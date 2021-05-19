//
//  datasource.swift
//  MetalNNDemo
//
//  Created by Tao Zhang on 2021/5/14.
//

import Foundation
import MetalPerformanceShaders

class ConvDataSource : NSObject, MPSCNNConvolutionDataSource {
    private let _descriptor : MPSCNNConvolutionDescriptor
    private let _device : MTLDevice
    private let _label : String
    private let _dataType = MPSDataType.float32
    private let _weightSize : Int
    private let _biasSize : Int
    private let _weightVector : MPSVector
    private let _weightMomentumVector: MPSVector
    private let _weightVelocityVector: MPSVector
    private let _biasVector : MPSVector
    private let _biasMomentumVector: MPSVector
    private let _biasVelocityVector: MPSVector
    private let _weightsAndBiasesState : MPSCNNConvolutionWeightsAndBiasesState
    private let _commandQueue : MTLCommandQueue
    private let _optimizer : ConvAdamOptimizer
    
    init(kernelWidth: Int,
         kernelHeight: Int,
         inputFeatureChannels: Int,
         outputFeatureChannels: Int,
         stride: Int,
         label: String,
         device: MTLDevice,
         commandQueue: MTLCommandQueue) {
        
        // Class members
        self._label = label
        self._device = device
        
        // Command queue
        self._commandQueue = commandQueue
        
        // Make descriptor
        _descriptor = MPSCNNConvolutionDescriptor(kernelWidth: kernelWidth,
                                                  kernelHeight: kernelHeight,
                                                  inputFeatureChannels: inputFeatureChannels,
                                                  outputFeatureChannels: outputFeatureChannels)
        _descriptor.strideInPixelsX = stride
        _descriptor.strideInPixelsY = stride
        _descriptor.fusedNeuronDescriptor = MPSNNNeuronDescriptor.cnnNeuronDescriptor(with: .none)
        // Allocate memory for vectors
        let weightsLength = inputFeatureChannels * kernelHeight * kernelWidth * outputFeatureChannels
        _weightSize = weightsLength * MemoryLayout<Float32>.size

        let weightVectorDescriptor = MPSVectorDescriptor(length: weightsLength,
                                                         dataType: _dataType);
        
        _weightVector = MPSVector(device: device,
                                  descriptor: weightVectorDescriptor)
        
        _weightMomentumVector = MPSVector(device: device,
                                          descriptor: weightVectorDescriptor)
        
        _weightVelocityVector = MPSVector(device: device,
                                          descriptor: weightVectorDescriptor)
        
        
        
        let biasLength = outputFeatureChannels
        _biasSize = biasLength * MemoryLayout<Float32>.size
        
        let biasVectorDescriptor = MPSVectorDescriptor(length: biasLength,
                                                       dataType: _dataType);
        
        _biasVector = MPSVector(device: device,
                                descriptor: biasVectorDescriptor)
        
        _biasMomentumVector = MPSVector(device: device,
                                        descriptor: biasVectorDescriptor)
        
        _biasVelocityVector = MPSVector(device: device,
                                        descriptor: biasVectorDescriptor)
        
        _weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(
            weights: _weightVector.data,
            biases: _biasVector.data)
        
        // Optimizer
        _optimizer = ConvAdamOptimizer(device: device,
                                       commandQueue: commandQueue)
        
        // Initialize the super class
        super.init()
        
        // Initialize values
        C_fillFloat32Array(_biasVector.data.contents(), 1.0, biasLength)
        resetRandom()
    }
    
    func dataType() -> MPSDataType {
        return _dataType
    }
    
    func descriptor() -> MPSCNNConvolutionDescriptor {
        return self._descriptor
    }
    
    func weights() -> UnsafeMutableRawPointer {
        return _weightVector.data.contents()
    }
    
    func biasTerms() -> UnsafeMutablePointer<Float>? {
        return UnsafeMutablePointer<Float>(OpaquePointer(_biasVector.data.contents()))
    }
    
    func load() -> Bool {
        let commandBuffer = MPSCommandBuffer(from: _commandQueue)
        _weightsAndBiasesState.synchronize(on: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        NSLog("ConvDataSource %@ is loaded in the GPU.", _label)
        return true
    }
    
    func purge() {}
    
    func label() -> String? {
        return self._label
    }
    
    func copy(with zone: NSZone? = nil) -> Any {
        // Not implemented
        NSLog("ConvDataSource.copy() is called, which is not implemented.")
        return self
    }
    
    func update(with commandBuffer: MTLCommandBuffer,
           gradientState: MPSCNNConvolutionGradientState,
           sourceState: MPSCNNConvolutionWeightsAndBiasesState) -> MPSCNNConvolutionWeightsAndBiasesState? {
        _optimizer.updater.encode(commandBuffer: commandBuffer,
                                  convolutionGradientState: gradientState,
                                  convolutionSourceState: sourceState,
                                  inputMomentumVectors: [_weightMomentumVector, _biasMomentumVector],
                                  inputVelocityVectors: [_weightVelocityVector, _biasVelocityVector],
                                  maximumVelocityVectors: nil,
                                  resultState: _weightsAndBiasesState)
        
        return _weightsAndBiasesState
    }
    
    // Reset all vectors to random number
    func resetRandom() {
        let randomDesc = MPSMatrixRandomDistributionDescriptor.uniformDistributionDescriptor(withMinimum: -0.2,
                                                                                             maximum: 0.2)
        
        let randomKernel = MPSMatrixRandomMTGP32(device: _device,
                                                 destinationDataType: .float32,
                                                 seed: 0,
                                                 distributionDescriptor: randomDesc)
        
        let commandBuffer = MPSCommandBuffer(from: _commandQueue)
        
        randomKernel.encode(commandBuffer: commandBuffer,
                            destinationVector: _weightVector)
        
        // Transfer data from the GPU to the CPU (will be a no-op on embedded GPUs)
        #if os(macOS)
        _weightVector.synchronize(on: commandBuffer)
        #endif
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Inform the GPU that the CPU has modified a section of the buffer.
        _weightMomentumVector.data.didModifyRange(0 ..< _weightSize)
        _weightVelocityVector.data.didModifyRange(0 ..< _weightSize)
        _weightVector.data.didModifyRange(0 ..< _weightSize)
        
        _biasMomentumVector.data.didModifyRange(0 ..< _biasSize)
        _biasVelocityVector.data.didModifyRange(0 ..< _biasSize)
        _biasVector.data.didModifyRange(0 ..< _biasSize)
    }
}


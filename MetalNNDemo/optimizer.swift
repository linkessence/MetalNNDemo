//
//  optimizer.swift
//  MetalNNDemo
//
//  Created by Tao Zhang on 2021/5/16.
//

import Foundation
import MetalPerformanceShaders

let DEFAULT_LEARNING_RATE           : Float32 = 1e-3
let DEFAULT_BETA1                   : Float64 = 0.9
let DEFAULT_BETA2                   : Float64 = 0.999
let DEFAULT_EPSILON                 : Float32 = 1e-8
let DEFAULT_GRADIENT_RESCALE        : Float32 = 1.0
let DEFAULT_REGULARIZATION_TYPE               = MPSNNRegularizationType.None
let DEFAULT_REGULARIZATION_SCALE    : Float32 = 1.0

class ConvAdamOptimizer : NSObject {
    
    let updater : MPSNNOptimizerAdam

    convenience init(device: MTLDevice) {
        self.init(learningRate: DEFAULT_LEARNING_RATE,
                  device: device)
    }
    
    convenience init(learningRate : Float32,
         device: MTLDevice) {
        self.init(learningRate : learningRate,
                  beta1: DEFAULT_BETA1,
                  beta2: DEFAULT_BETA2,
                  epsilon: DEFAULT_EPSILON,
                  device : device)
    }
    
    init(learningRate: Float32,
         beta1: Float64,
         beta2: Float64,
         epsilon: Float32,
         device: MTLDevice) {
        let desc = MPSNNOptimizerDescriptor(learningRate: learningRate,
                                            gradientRescale: DEFAULT_GRADIENT_RESCALE,
                                            regularizationType: DEFAULT_REGULARIZATION_TYPE,
                                            regularizationScale: DEFAULT_REGULARIZATION_SCALE)
        
        updater = MPSNNOptimizerAdam(device: device,
                                      beta1: beta1,
                                      beta2: beta2,
                                      epsilon: epsilon,
                                      timeStep: 0,
                                      optimizerDescriptor: desc)
        super.init()
    }
}

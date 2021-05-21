//
//  optimizer.swift
//  MetalNNDemo
//
//  Created by Tao Zhang on 2021/5/16.
//

import Foundation
import MetalPerformanceShaders

let DEFAULT_LEARNING_RATE           : Float32 = 1e-3
let DEFAULT_GRADIENT_RESCALE        : Float32 = 1.0
let DEFAULT_REGULARIZATION_TYPE               = MPSNNRegularizationType.None
let DEFAULT_REGULARIZATION_SCALE    : Float32 = 1.0
let DEFAULT_MOMENTUM_SCALE          : Float32 = 0.0

class ConvSGDOptimizer : NSObject {
    
    let updater : MPSNNOptimizerStochasticGradientDescent

    convenience init(device: MTLDevice) {
        self.init(learningRate: DEFAULT_LEARNING_RATE,
                  device: device)
    }
    
    init(learningRate : Float32,
         device: MTLDevice) {
        let desc = MPSNNOptimizerDescriptor(learningRate: learningRate,
                                            gradientRescale: DEFAULT_GRADIENT_RESCALE,
                                            regularizationType: DEFAULT_REGULARIZATION_TYPE,
                                            regularizationScale: DEFAULT_REGULARIZATION_SCALE)
        
        updater = MPSNNOptimizerStochasticGradientDescent(device: device,
                                                          momentumScale: DEFAULT_MOMENTUM_SCALE,
                                                          useNesterovMomentum: false,
                                                          optimizerDescriptor: desc)
        super.init()
    }
}

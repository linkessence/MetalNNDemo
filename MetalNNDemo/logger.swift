//
//  logger.swift
//  MetalNNDemo
//
//  Created by Tao Zhang on 2021/5/12.
//

import Foundation

func initLogger() {
    C_registerLoggerCallback({message in
        if (message != nil) {
            NSLog("%s", message!)
        }
    })
}

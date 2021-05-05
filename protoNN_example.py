# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import print_function
import sys
import os
import numpy as np
from edgeml_pytorch.trainer.protoNNTrainer import ProtoNNTrainer
from edgeml_pytorch.graph.protoNN import ProtoNN
import edgeml_pytorch.utils as utils
import helpermethods as helper
import torch
import time

def main(train, test, device, proj_dim, num_proj):
    # print("Train:", train.shape)
    # print(train)
    # print("Test:", test.shape)
    # print(test)

    PROJECTION_DIM = proj_dim
    NUM_PROTOTYPES = num_proj
    print(PROJECTION_DIM, NUM_PROTOTYPES)
    REG_W = 0.0
    REG_B = 0.0
    REG_Z = 0.0
    SPAR_W = 1.0
    SPAR_B = 1.0
    SPAR_Z = 1.0
    LEARNING_RATE = 0.1
    NUM_EPOCHS = 15
    BATCH_SIZE = 32
    PRINT_STEP = 200
    VAL_STEP = 5
    OUT_DIR = './'
    GAMMA = 0.0015

    # Load data
    x_train, y_train = train[:, 1:], train[:, 0]
    x_test, y_test = test[:, 1:], test[:, 0]
    # Convert y to one-hot
    minval = min(min(y_train), min(y_test))
    numClasses = max(y_train) - min(y_train) + 1
    numClasses = max(numClasses, max(y_test) - min(y_test) + 1)
    numClasses = int(numClasses)
    y_train = helper.to_onehot(y_train, numClasses, minlabel=minval)
    y_test = helper.to_onehot(y_test, numClasses, minlabel=minval)
    dataDimension = x_train.shape[1]

    W, B, gamma = helper.getGamma(GAMMA, PROJECTION_DIM, dataDimension,
                                  NUM_PROTOTYPES, x_train)

    # Setup input and train protoNN
    protoNN = ProtoNN(dataDimension, PROJECTION_DIM,
                      NUM_PROTOTYPES, numClasses,
                      gamma, W=W, B=B).to(device)

    trainer = ProtoNNTrainer(protoNN, REG_W, REG_B, REG_Z,
                             SPAR_W, SPAR_B, SPAR_Z,
                             LEARNING_RATE, lossType='xentropy', device=device)
    # Train the protoNN object
    train_start_ts = time.time()
    trainer.train(BATCH_SIZE, NUM_EPOCHS, x_train, x_test,
                  y_train, y_test, printStep=PRINT_STEP, valStep=VAL_STEP)
    train_end_ts = time.time()

    # Print some summary metrics
    test_start_ts = time.time()
    x_, y_= (torch.Tensor(x_test)).to(device), (torch.Tensor(y_test)).to(device)

    logits = protoNN.forward(x_)
    _, predictions = torch.max(logits, dim=1)
    _, target = torch.max(y_, dim=1)
    acc, count = trainer.accuracy(predictions, target)

    test_end_ts = time.time()

    #Model needs to be on cpu for numpy operations below
    protoNN = protoNN.cpu()
    W, B, Z, gamma  = protoNN.getModelMatrices()
    matrixList = [W, B, Z]
    matrixList = [x.detach().numpy() for x in matrixList]
    sparcityList = [SPAR_W, SPAR_B, SPAR_Z]
    nnz, size, sparse = helper.getModelSize(matrixList, sparcityList)
    print("Train data shape:", train.shape, "Test:", test.shape)
    print("Final test accuracy:", acc, "Training time:", train_end_ts-train_start_ts, "Testing time:", test_end_ts-test_start_ts)
    print("Model size constraint (Bytes): ", size)
    print("Number of non-zeros: ", nnz)
    nnz, size, sparse = helper.getModelSize(matrixList, sparcityList,
                                            expected=False)
    print("Actual model size: ", size)
    print("Actual non-zeros: ", nnz)
    print("Saving model matrices to: ", OUT_DIR)
    np.save(OUT_DIR + '/W.npy', matrixList[0])
    np.save(OUT_DIR + '/B.npy', matrixList[1])
    np.save(OUT_DIR + '/Z.npy', matrixList[2])
    np.save(OUT_DIR + '/gamma.npy', gamma)

    return acc


if __name__ == '__main__':
    main()

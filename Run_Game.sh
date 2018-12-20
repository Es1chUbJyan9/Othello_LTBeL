#!/bin/bash

FILE="./data/model.pb"

    if [ -f $FILE ]
        then
        bazel build -c opt --config=cuda --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.1 --copt=-msse4.2 --config=mkl --copt="-D CNN" //Othello_LTBeL/src/Othello_LTBeL:Othello_LTBeL
        
        else
        bazel build -c opt --config=cuda --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.1 --copt=-msse4.2 --config=mkl //Othello_LTBeL/src/Othello_LTBeL:Othello_LTBeL
        fi

../bazel-bin/Othello_LTBeL/src/Othello_LTBeL/Othello_LTBeL

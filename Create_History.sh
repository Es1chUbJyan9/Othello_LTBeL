#!/bin/bash

for loop in $(seq 1 2500) 
do
    FILE="./data/model.pb"
    if [ -f $FILE ]
    then
        bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.1 --copt=-msse4.2 --config=mkl --copt="-D CNN" //Othello_LTBeL/src/Othello_LTBeL_Self:Othello_LTBeL_Self
    else
        bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.1 --copt=-msse4.2 --config=mkl //Othello_LTBeL/src/Othello_LTBeL_Self:Othello_LTBeL_Self
    fi

    ../bazel-bin/Othello_LTBeL/src/Othello_LTBeL_Self/Othello_LTBeL_Self
    
    echo "\n##########################"
    echo "Loop: ${loop} is Compelet."
    echo "##########################\n"

done

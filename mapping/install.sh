#!/bin/bash

cd third_party/sparse_octree
pip3 install .

cd ../sparse_voxels
pip3 install .

cd ../tiny-cuda-nn_H3_mapping
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
cd bindings/torch
pip3 install .

cd ../ELSED_H3_mapping
pip3 install .

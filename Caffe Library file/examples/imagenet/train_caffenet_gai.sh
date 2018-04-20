TOOLS=./build/tools
GLOG_logtostderr=0 GLOG_log_dir=models/bvlc_reference_caffenet/Log/ \
./build/tools/caffe train \
    --solver=models/bvlc_reference_caffenet/solver.prototxt

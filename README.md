## nomadrt (real time)
implementation of NoMAD with tensorrt for running on edge devices.

### convert from pytorch to onnx
```bash
python onnx_export.py -i nomad.pth -t config/nomad.yaml -o weights
```

### convert onnx models to tensorrt's engine
```bash
cd weights
trtexec --onnx=action.onnx --best --useCudaGraph --saveEngine=action.engine
trtexec --onnx=encoder.onnx --best --useCudaGraph --saveEngine=encoder.engine
trtexec --onnx=distance.onnx --best --useCudaGraph --saveEngine=distance.engine
```

### compile the ros2 package
```bash
colcon build --symlink-install
```

## special thanks
- https://github.com/robodhruv/visualnav-transformer ground-breaking work with creating the foundational model
- https://github.com/UT-ADL/milrem_visual_offroad_navigation onnx implementation and code organization
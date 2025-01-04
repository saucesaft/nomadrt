import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

class EncoderModuleTRT:
    def __init__(self, engine_file_path, logger):

        self.ros_logger = logger
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        trt.init_libnvinfer_plugins(self.logger, "")

        # Load the TensorRT engine
        self.engine = self._load_engine(engine_file_path)
        self.context = self.engine.create_execution_context()

        # Allocate memory for inputs and outputs
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

    def _load_engine(self, engine_file_path):
        with open(engine_file_path, "rb") as f:
            engine_data = f.read()
        return self.runtime.deserialize_cuda_engine(engine_data)

    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to bindings
            bindings.append(int(device_mem))
            # Append to the appropriate list
            if self.engine.binding_is_input(binding):
                inputs.append({"host": host_mem, "device": device_mem})
            else:
                outputs.append({"host": host_mem, "device": device_mem})

        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        # Copy input data to the device
        np.copyto(self.inputs[0]["host"], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)

        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy output data back to the host
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()

        return np.array(self.outputs[0]["host"])
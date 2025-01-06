import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

class EncoderModuleTRT:
    def __init__(self, engine_file_path, config):

        self.config = config
        self.ros_logger = config['logger']

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        trt.init_libnvinfer_plugins(self.logger, "")

        # load the TensorRT engine
        self.engine = self._load_engine(engine_file_path)
        self.context = self.engine.create_execution_context()

        # allocate memory for inputs and outputs
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

    def _load_engine(self, engine_file_path):
        with open(engine_file_path, "rb") as f:
            engine_data = f.read()
        return self.runtime.deserialize_cuda_engine(engine_data)

    def _allocate_buffers(self):
        inputs = {}
        outputs = {}
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # append the device buffer to bindings
            bindings.append(int(device_mem))

            # append to the appropriate list
            if self.engine.binding_is_input(binding):                
                inputs[binding] = {
                            "host": host_mem,
                            "device": device_mem,
                            "shape": self.engine.get_binding_shape(binding),
                            "type": trt.nptype(self.engine.get_binding_dtype(binding))
                        }
            else:
                outputs[binding] = {
                            "host": host_mem,
                            "device": device_mem,
                            "shape": self.engine.get_binding_shape(binding),
                            "type": trt.nptype(self.engine.get_binding_dtype(binding))
                        }

        return inputs, outputs, bindings, stream

    def encode_features(self, obs, goal, mask):
        # copy input data to the device
        np.copyto(self.inputs['obs_img']["host"], obs.ravel())
        cuda.memcpy_htod_async(
            self.inputs['obs_img']["device"],
            self.inputs['obs_img']["host"],
            self.stream
        )

        np.copyto(self.inputs['goal_img']["host"], goal.ravel())
        cuda.memcpy_htod_async(
            self.inputs['goal_img']["device"],
            self.inputs['goal_img']["host"],
            self.stream
        )

        np.copyto(self.inputs['input_goal_mask']["host"], mask.ravel())
        cuda.memcpy_htod_async(
            self.inputs['input_goal_mask']["device"],
            self.inputs['input_goal_mask']["host"],
            self.stream
        )

        # run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # copy output data back to the host
        cuda.memcpy_dtoh_async(self.outputs['4022']["host"], self.outputs['4022']["device"], self.stream)
        self.stream.synchronize()

        return np.array(self.outputs['4022']["host"])
# io deps
import os

# ml deps
import numpy as np
import tensorrt as trt

from nomadrt.encoder_trt import EncoderModuleTRT
from nomadrt.action_trt import ActionModuleTRT
from nomadrt.distance_trt import DistanceModuleTRT
from nomadrt.model.noise_scheduler import DDPMScheduler
from nomadrt.model.nomad_util import get_action

class NomadTRT:
    def __init__(self, model_path, logger, num_samples=10, len_traj_pred=8, num_diffusion_iters=10):

        self.ros_logger = logger

        self.config = {
            "logger": self.ros_logger,
            "num_samples": num_samples,
            "len_traj_pred": len_traj_pred,
            "num_diffusion_iters": num_diffusion_iters,
        }

        self.encoder_session = EncoderModuleTRT( os.path.join(model_path, 'encoder.engine'), self.config )
        self.dist_session = DistanceModuleTRT( os.path.join(model_path, 'distance.engine'), self.config )
        self.action_session = ActionModuleTRT( os.path.join(model_path, 'action.engine'), self.config )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
        )

    def predict(self, obs, goal):
        vision_features = self.encoder_session.encode_features(obs, goal)
        predicted_distance = self.dist_session.predict_distance(vision_features)
        predicted_actions = self.action_session.predict_actions(vision_features)
        return predicted_distance, predicted_actions
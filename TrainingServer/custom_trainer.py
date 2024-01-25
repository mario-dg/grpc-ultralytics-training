from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG


class CustomDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.progress_queue = overrides.pop("progress_queue", None)
        super().__init__(cfg, overrides, _callbacks)

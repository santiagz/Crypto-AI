import os
from loguru import logger


checkpoint_path = "data/checkpoints/point_1.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
logger.success(f"checkpoint_dir: {checkpoint_dir}")

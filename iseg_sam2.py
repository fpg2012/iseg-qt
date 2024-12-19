import iseg_qt
import numpy as np
import json
import argparse
import sys
sys.path.append("../sam2")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class Sam2PredictorAdapter(iseg_qt.Predictor):
    
    def __init__(self, sam_predictor: SAM2ImagePredictor):
        self.image = None
        self.prev_masks = None
        self.sam_predictor: SAM2ImagePredictor = sam_predictor
        self.first_click = True
    
    def set_image(self, image: np.ndarray):
        self.image = image
        self.sam_predictor.set_image(image)
        self.first_click = True
    
    def predict(self, point_coords, point_labels):
        point_coords = [[int(p[0]), int(p[1])] for p in point_coords]
        mask = None
        if self.first_click:
            masks, _, logits = self.sam_predictor.predict(
                point_coords=np.array(point_coords),
                point_labels=np.array(point_labels),
                multimask_output=True
            )
            mask = masks[0]
            self.prev_masks = logits[0]
            self.first_click = False
        else:
            masks, _, logit = self.sam_predictor.predict(
                point_coords=np.array(point_coords),
                point_labels=np.array(point_labels),
                multimask_output=False,
                mask_input=self.prev_masks[None, :, :]
            )
            mask = masks[0]
            self.prev_masks = logit[0]
        return mask

    def reset(self):
        self.image = None
        self.prev_masks = None
        self.first_click = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_config', type=str)
    args = parser.parse_args()
    
    device = args.device
    checkpoint = args.checkpoint
    model_config = args.model_config

    if device == 'xpu':
        import torch
        import intel_extension_for_pytorch as ipex
    
    sam = build_sam2(model_config, checkpoint, device, mode='eval')
    sam_predictor = SAM2ImagePredictor(sam)

    predictor = Sam2PredictorAdapter(sam_predictor=sam_predictor)

    iseg_qt.run_application(predictor=predictor)


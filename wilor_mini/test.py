import torch
import cv2
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float16

pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype,verbose = True)
img_path = "assets/img.png"
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
outputs = pipe.predict(image) # outputs is a list of dictionaries(the len is the num of hands)
out = outputs[1]    

"""
out keys: ['hand_bbox', 'is_right', 'wilor_preds']

out['wilor_preds'] keys: 
['global_orient', 'hand_pose', 'betas', 'pred_cam', 'pred_keypoints_3d', 
'pred_vertices', 'pred_cam_t_full', 'scaled_focal_length', 'pred_keypoints_2d']
"""
is_right = out['is_right'] # float, 1 for right hand, 0 for left hand

verts = out["wilor_preds"]['pred_vertices'] # (1, 778, 3)

pred_keypoints_3d = out["wilor_preds"]["pred_keypoints_3d"] # (1, 21, 3)
hand_pose = out["wilor_preds"]['hand_pose'] # (1, 15, 3)

cam_t = out["wilor_preds"]['pred_cam_t_full'] # (1, 3)
pred_cam = out["wilor_preds"]['pred_cam'] # (1, 3)
# scaled_focal_length = out["wilor_preds"]['scaled_focal_length']

global_orient = out["wilor_preds"]['global_orient'] # (1,1,3)

betas = out["wilor_preds"]['betas'] # (1, 10)




"""

cam_t: [[   -0.12962    0.061737      4.6129]]
global_orient: [[[     1.2529      1.4941      1.8721]]]
hand_pose: (1, 15, 3)
pred_cam: [[     4.0859    -0.13708    0.061737]]
pred_keypoints_3d: [   0.095459   0.0063629    0.006176]

"""
import torch
import numpy as np
from ultralytics import YOLO
from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
import cv2
from wilor.utils.renderer import Renderer, cam_crop_to_full
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

OPERATOR2MANO = np.array(
    [
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ]
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
    (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
]

class HandDetector:
    def __init__(self, hand_type="Right"):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dtype = torch.float16
        self.pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
        self.detected_hand_type = hand_type
        self.operator2mano = OPERATOR2MANO

    def detect(self, bgr_image: np.array):
        """
        Detect hands and estimate their 3D pose using YOLO + WiLoR.
        
        Args:
            rgb_image (np.array): Input image in RGB format (H, W, 3).
            
        Returns:
            num_hands (int): Number of detected hands.
            joint_positions (np.array): 3D positions of the hand joints.
            keypoints_2d (np.array): 2D keypoints of the detected hands.
        """
        
        outputs = self.pipe.predict(bgr_image)
        is_detected_hand = 0
        joints = None
        pred_keypoints_2d = None

        if len(outputs) == 0:
            return 0, None, None

        for out in outputs:
            is_right = out['is_right']  # float, 1 for right hand, 0 for left hand
            if (is_right == 1 and self.detected_hand_type == "Left") or (is_right == 0 and self.detected_hand_type == "Right"):
                continue
            else:
                pred_keypoints_2d = out["wilor_preds"]["pred_keypoints_2d"][0]  # (1, 21, 2) -> (21, 2)
                pred_keypoints_3d = out["wilor_preds"]["pred_keypoints_3d"][0] # (1, 21, 3) -> (21, 3)
                pred_cam_t_full = out["wilor_preds"]['pred_cam_t_full'][0] # (1, 3) -> (3,)
                # print("pred_cam_t_full",pred_cam_t_full)
                pred_cam_t_full[2] = pred_cam_t_full[2] -0.6
                pred_cam_t_full[1] = pred_cam_t_full[1] -0.2 
                pred_keypoints_3d = pred_keypoints_3d + pred_cam_t_full
                joints = pred_keypoints_3d @ self.operator2mano.T
                is_detected_hand = 1
                break

        return is_detected_hand, joints, pred_keypoints_2d


    @staticmethod
    def draw_skeleton_on_image(image, keypoints_2d, style="default"):
    
        if keypoints_2d is None:
            return image

        # 复制图像，避免修改原始图像
        img_copy = image.copy()
        
        # 设置关键点和线条颜色
        if style == "white":
            point_color = (255, 255, 255)  # 白色
            line_color = (255, 255, 255)   # 白色
        else:
            point_color = (0, 0, 255)  # 红色
            line_color = (255, 0, 0)   # 蓝色

        # 画骨架连接线
        for connection in HAND_CONNECTIONS:
            pt1 = tuple(keypoints_2d[connection[0]].astype(int))
            pt2 = tuple(keypoints_2d[connection[1]].astype(int))
            cv2.line(img_copy, pt1, pt2, line_color, 2)

        # 画关键点
        for i, keypoint in enumerate(keypoints_2d):
            x, y = int(keypoint[0]), int(keypoint[1])
            cv2.circle(img_copy, (x, y), radius=4, color=point_color, thickness=-1)

        return img_copy
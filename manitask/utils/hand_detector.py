import torch
import numpy as np
from ultralytics import YOLO
import cv2
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline


OPERATOR2MANO = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)

HAND_ROTATE = np.array(
    [
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
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
    def __init__(self, hand_type="Right",trans_scale=1.0):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dtype = torch.float16
        self.pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
        self.detected_hand_type = hand_type
        self.operator2mano = OPERATOR2MANO
        self.init_position = None
        self.keypoints_2d = None
        self.origin_2d = None
        self.axes_2d = None
        self.trans_scale = trans_scale
        

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
        points = None
        

        if len(outputs) == 0:                
            return 0, None

        for out in outputs:
            # self.detected_hand_type = "right"
            is_right = out['is_right']  # float, 1 for right hand, 0 for left hand
            if (is_right == 1 and self.detected_hand_type == "left") or (is_right == 0 and self.detected_hand_type == "right"):
                continue
            else:
                self.keypoints_2d = out["wilor_preds"]["pred_keypoints_2d"][0]  # (1, 21, 2) -> (21, 2)
                pred_keypoints_3d = out["wilor_preds"]["pred_keypoints_3d"][0] # (1, 21, 3) -> (21, 3)
                pred_cam_t_full = out["wilor_preds"]['pred_cam_t_full'][0] # (1, 3) -> (3,)

                if self.init_position is None:
                    self.init_position = pred_cam_t_full

                cam_t_full = pred_cam_t_full - self.init_position


                pred_keypoints_3d = pred_keypoints_3d + self.trans_scale * cam_t_full
                points = pred_keypoints_3d @ self.operator2mano.T
    
                is_detected_hand = 1
                
                break


        return is_detected_hand, points


    def draw_skeleton_on_image(self, image, style="default"):
        if self.keypoints_2d is None:
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
            pt1 = tuple(self.keypoints_2d[connection[0]].astype(int))
            pt2 = tuple(self.keypoints_2d[connection[1]].astype(int))
            cv2.line(img_copy, pt1, pt2, line_color, 2)

        # 画关键点
        for i, keypoint in enumerate(self.keypoints_2d):
            x, y = int(keypoint[0]), int(keypoint[1])
            cv2.circle(img_copy, (x, y), radius=4, color=point_color, thickness=-1)
        
            
        return img_copy
    

    @staticmethod
    def project_full_img(points, cam_trans, focal_length=512):
        # img size: 640,480
        camera_center = [320, 240]
        K = torch.eye(3) 
        K[0,0] = focal_length
        K[1,1] = focal_length
        K[0,2] = camera_center[0]
        K[1,2] = camera_center[1]
        points = points + cam_trans
        points = points / points[..., -1:] 
        
        V_2d = (K @ points.T).T 
        return V_2d[..., :-1]
    

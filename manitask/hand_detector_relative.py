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
    def __init__(self, hand_type="Right"):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dtype = torch.float16
        self.pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
        self.detected_hand_type = hand_type
        self.operator2mano = OPERATOR2MANO
        self.init_position = None
        self.translation = None
        self.keypoints_2d = None
        self.origin_2d = None
        self.axes_2d = None
        

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
        # pred_keypoints_2d = None

        if len(outputs) == 0:
            return 0, None

        for out in outputs:
            # self.detected_hand_type = "right"
            is_right = out['is_right']  # float, 1 for right hand, 0 for left hand
            if (is_right == 1 and self.detected_hand_type == "Left") or (is_right == 0 and self.detected_hand_type == "Right"):
                continue
            else:
                self.keypoints_2d = out["wilor_preds"]["pred_keypoints_2d"][0]  # (1, 21, 2) -> (21, 2)
                pred_keypoints_3d = out["wilor_preds"]["pred_keypoints_3d"][0] # (1, 21, 3) -> (21, 3)
                pred_cam_t_full = out["wilor_preds"]['pred_cam_t_full'][0] # (1, 3) -> (3,)
                # print("pred_cam_t_full",pred_cam_t_full)
                
                # pred_cam_t_full[2] = pred_cam_t_full[2] -0.8
                # pred_cam_t_full[1] = pred_cam_t_full[1] -0.1
                # print(pred_cam_t_full)
                if self.init_position is None:
                    self.init_position = pred_cam_t_full
                    self.origin_2d = self.keypoints_2d[0]
                       # 定义三个坐标轴的向量（相对于原点）
                    axis_length = 0.1  # 轴的长度，可以根据需要调整
                    origin_3d = pred_keypoints_3d[0]  # 以手腕为原点
                    
                    axes_3d = np.zeros((3, 3))
                    # 创建包含原点和三个轴端点的矩阵
                    axes_3d[0] = origin_3d + np.array([axis_length, 0, 0])  # x轴
                    axes_3d[1] = origin_3d + np.array([0, axis_length, 0])  # y轴
                    axes_3d[2] = origin_3d + np.array([0, 0, axis_length])  # z轴
    
                    # 转换为torch张量并投影
                    axes_3d_t = torch.tensor(axes_3d, dtype=torch.float32)
                    cam_trans = torch.tensor(pred_cam_t_full, dtype=torch.float32)

                    self.axes_2d = self.project_full_img(axes_3d_t, cam_trans).numpy()
                    

                delta_cam_t_full = pred_cam_t_full - self.init_position

                # 为三个轴分别设置过滤参数
                thresholds = np.array([0.01, 0.01, 0.01])  # X, Y, Z轴的阈值，可以分别调整
                max_deltas = np.array([0.05, 0.05, 0.05])    # X, Y, Z轴的最大变化幅度

                # 对各轴分别进行阈值过滤
                for i in range(3):
                    # 小于阈值的变化置零
                    if abs(delta_cam_t_full[i]) < thresholds[i]:
                        delta_cam_t_full[i] = 0
                    else:
                        # 限制最大变化幅度
                        delta_cam_t_full[i] = np.clip(delta_cam_t_full[i], -max_deltas[i], max_deltas[i])

                # print("delta_cam",delta_cam_t_full)
                if self.translation is None:
                    self.translation = delta_cam_t_full
                else:
                    self.translation = self.translation + 0.2*delta_cam_t_full
                pred_keypoints_3d = pred_keypoints_3d + self.translation
                joints = pred_keypoints_3d @ self.operator2mano.T

                joints[:,2] = joints[:,2] + 0.08 # the z position of the wrist you want
                joints[:,0] = joints[:,0] + 0.25 # the x position of the wrist you want
                # print(joints[0])
                

                
                # if self.init_wrist_position is None:
                #     self.init_wrist_position = joints[0]
                # joints = joints - self.init_wrist_position

                is_detected_hand = 1
                break

        return is_detected_hand, joints


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
        
        # 标记 origin_2d 点并绘制坐标轴
        if self.origin_2d is not None:
            origin = tuple(self.origin_2d.astype(int))
            axis_x = tuple(self.axes_2d[0].astype(int))
            axis_y = tuple(self.axes_2d[1].astype(int))
            axis_z = tuple(self.axes_2d[2].astype(int))
            # 将 origin_2d 标记为较大的圆，使用不同颜色
            cv2.circle(img_copy, origin, radius=10, color=(0, 255, 255), thickness=-1)
            # 绘制坐标轴
            # 绘制三个坐标轴
            cv2.line(img_copy, origin, axis_x, (0, 0, 255), 2)  # X轴 - 红色
            # cv2.putText(img_copy, "X", self.axes_2d[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.line(img_copy, origin, axis_y, (0, 255, 0), 2)  # Y轴 - 绿色
            # cv2.putText(img_copy, "Y", self.axes_2d[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.line(img_copy, origin, axis_z,(255, 0, 0), 2)  # Z轴 - 蓝色
            # cv2.putText(img_copy, "Z", self.axes_2d[2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
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
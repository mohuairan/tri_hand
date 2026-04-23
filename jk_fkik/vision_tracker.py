import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, max_hands=1, detection_con=0.7, tracking_con=0.7):
        """
        初始化 MediaPipe 手部骨架追踪器
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_con,
            min_tracking_confidence=tracking_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, img):
        """
        处理单帧图像，返回骨架标注后的图像与关键点 3D 坐标
        :param img: OpenCV BGR 图像
        :return: (标注后的 img, 关键点坐标数组 np.ndarray(21, 3)) 
                 如果未检测到手，坐标数组返回 None
        """
        # MediaPipe 需要 RGB 格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        landmarks_3d = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制 2D 骨架连线到画面上
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # 提取 3D 坐标 (21个关键点)
                # 注意：x, y, z 都是归一化的值，z 代表相对手腕的深度
                landmarks_list = []
                for lm in hand_landmarks.landmark:
                    landmarks_list.append([lm.x, lm.y, lm.z])
                landmarks_3d = np.array(landmarks_list)
                
                # 这里只取检测到的第一只手
                break
                
        return img, landmarks_3d

def main():
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头！")
        return
        
    print("开始骨架捕捉。请将手移动到摄像头前。按 'q' 退出。")
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        # 镜像翻转画面，符合直觉
        img = cv2.flip(img, 1)
        
        img, landmarks = tracker.process_frame(img)
        
        if landmarks is not None:
            # 在画面左上角提示已捕捉
            cv2.putText(img, "Hand Detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 您可以在这里打印特定关键点，比如指尖：
            # index_tip = landmarks[8]
            # print(f"食指指尖 3D: {index_tip}")
                        
        cv2.imshow("Hand Tracking (MediaPipe)", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

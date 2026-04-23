import cv2
import threading
import time
import numpy as np
import mujoco
import mujoco.viewer

from vision_tracker import HandTracker
from retargeting import HandRetargeter

# 共享变量与锁
shared_ctrl_dict = {}
ctrl_lock = threading.Lock()
running = True

def vision_thread_func():
    global shared_ctrl_dict, running
    
    tracker = HandTracker()
    retargeter = HandRetargeter()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！视觉线程退出。")
        running = False
        return
        
    print(">>> 视觉跟踪与逆运动学线程已启动 <<<")
    
    while running:
        success, img = cap.read()
        if not success:
            continue
            
        img = cv2.flip(img, 1)
        img, landmarks_3d = tracker.process_frame(img)
        
        if landmarks_3d is not None:
            # 运行核心重定向与逆解计算
            ctrl_dict = retargeter.process(landmarks_3d)
            
            # 将计算好的关节角写入共享内存
            with ctrl_lock:
                shared_ctrl_dict.update(ctrl_dict)
                
            cv2.putText(img, "Tracking & Solving IK", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(img, "No Hand Detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
        cv2.imshow("Teleoperation Camera", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(">>> 视觉线程已退出 <<<")


def main():
    global shared_ctrl_dict, running
    
    # 加载 MuJoCo 模型
    xml_path = "mujoco_model/jack_hand.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 取消重力，防止模型不受控掉落
    model.opt.gravity[:] = 0

    # 预先获取所有的 actuator 名称和索引，加速控制回调
    actuator_dict = {}
    for i in range(model.nu):
        name = model.actuator(i).name
        actuator_dict[name] = i

    # MuJoCo 控制器回调函数 (由物理引擎以 1000Hz 极高频调用)
    def control_cb(m, d):
        with ctrl_lock:
            for name, rad_val in shared_ctrl_dict.items():
                if name in actuator_dict:
                    idx = actuator_dict[name]
                    # 限制在合法范围内
                    lo = m.actuator_ctrlrange[idx, 0]
                    hi = m.actuator_ctrlrange[idx, 1]
                    val = np.clip(rad_val, lo, hi)
                    d.ctrl[idx] = val

    # 注册回调
    mujoco.set_mjcb_control(control_cb)

    # 启动视觉后台线程
    v_thread = threading.Thread(target=vision_thread_func)
    v_thread.start()

    print(">>> 启动 MuJoCo Viewer <<<")
    # 启动官方 Viewer，阻塞在主线程
    try:
        mujoco.viewer.launch(model, data)
    except KeyboardInterrupt:
        pass
    finally:
        # Viewer 退出后，清理后台线程
        running = False
        v_thread.join()
        print("程序已完全退出。")

if __name__ == "__main__":
    main()

import pyrealsense2 as rs
import numpy as np
import cv2
import time

# 设置 RealSense 流
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB 流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 深度流

# 启动流
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align = rs.align(rs.stream.color)

# 文件保存路径
output_dir = './output'
rgb_png_output = f'{output_dir}/rgb'
rgb_jpg_output = f'{output_dir}/JPEGImages'
depth_output = f'{output_dir}/depth'

save_flag=False

# 创建保存路径
import os
os.makedirs(rgb_png_output, exist_ok=True)
os.makedirs(rgb_jpg_output, exist_ok=True)
os.makedirs(depth_output, exist_ok=True)

# 捕获帧
try:
    count=0
    while True:
        start_time = time.time()
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 提取帧数据
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 转换深度图至可视化格式
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 显示图像
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow('RealSense', images)

        # cv2.imshow('color_image', color_image)
        # cv2.imshow('depth_image', depth_image)

        # 存储图像
        if (save_flag):
            count+=1
            filename = f"{rgb_png_output}/{str(count).zfill(5)}.png"
            print(f"已保存: {filename}")
            cv2.imwrite(f"{rgb_png_output}/{str(count).zfill(5)}.png", color_image)
            cv2.imwrite(f"{rgb_jpg_output}/{str(count).zfill(5)}.jpg", color_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # 保存JPG格式的彩色图
            cv2.imwrite(f"{depth_output}/{str(count).zfill(5)}.png", depth_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('s'):
            print("start recording")
            save_flag=True

        end_time = time.time()
        # print(f"Frame Rate: {1/(end_time - start_time):.2f} FPS")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
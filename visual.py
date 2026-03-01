import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import mmrotate
import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector

config_file = r'C:\Users\dians\Desktop\mm\tools\work_dirs\rotated_retinanet-24-1\my.py'
checkpoint_file = r'C:\Users\dians\Desktop\mm\tools\work_dirs\rotated_retinanet-24-1\epoch_24.pth'
image_file = r'C:\Users\dians\Desktop\mm\data\test\images\IMG_20230507_084843.jpg'
output_file = 'result_with_angle.jpg'

model = init_detector(config_file, checkpoint_file, device='cuda:0')
result = inference_detector(model, image_file)

img = cv2.imread(image_file)
classes = ['green_fruit', 'flower']
colors = [(0, 255, 0), (0, 0, 255)]
arrow_len = 150  # 箭头长度，可以调大

for cls_id, bboxes in enumerate(result):
    for bbox in bboxes:
        cx, cy, w, h, angle, score = bbox
        angle_deg = np.degrees(angle)

        # 画旋转框
        rect = ((cx, cy), (w, h), angle_deg)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, colors[cls_id], 3)

        # 画箭头，方向就是angle
        start = (int(cx), int(cy))
        end = (int(cx + arrow_len * np.cos(angle)), int(cy + arrow_len * np.sin(angle)))
        cv2.arrowedLine(img, start, end, (0, 165, 255), 4, tipLength=0.3)

        # 文字标注
        label = f'{classes[cls_id]} {score:.2f} {angle_deg:.1f}deg'
        cv2.putText(img, label, (int(cx), int(cy) - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, colors[cls_id], 3)

cv2.imwrite(output_file, img)
print(f'保存到: {output_file}')
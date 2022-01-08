import numpy as np
import json

def get_distance_image(raw_image_path, inf_DP, distance_limit = 100):
    with open(raw_image_path, "rb") as f:
        disparity_image = f.read()
    img = []
    for d in range(0, len(disparity_image), 2):
        d_image = disparity_image[d]
        if d_image:
            d_image += disparity_image[d + 1] / 256.0
        img.append(d_image)
    img = np.array(img, dtype=np.float32)

    not_inf_index = img > inf_DP
    img[not_inf_index] = 560 / (img[not_inf_index] - inf_DP)
    #img[img > distance_limit] = 0
    img = img / distance_limit
    img = img.reshape(105, -1)
    img = np.flip(img, 0)
    img = img[:, :250]
    return img

def load_distance_image(file_path, inf_DP, distance_limit=100):

    raw_width = 256
    raw_height = 105

    with open(file_path, 'rb') as f:
        disparity_image = f.read()

    seisu_bu = np.frombuffer(disparity_image[0::2], dtype=np.uint8).astype(np.float32)
    shosu_bu = np.frombuffer(disparity_image[1::2], dtype=np.uint8).astype(np.float32) / 256.0

    disparity = seisu_bu + shosu_bu
    disparity = disparity.reshape((raw_height, raw_width))

    # 右側6画素を取り除く
    disparity = disparity[:, :raw_width - 6]

    # 上下反転
    disparity = disparity[::-1]

    not_inf_index = disparity > inf_DP

    disparity[disparity <= inf_DP] = 0
    disparity[not_inf_index] = 560.0 / (disparity[not_inf_index] - inf_DP)
    disparity = disparity / distance_limit
    return disparity

video_no = 10
frame_no = 50

disparity_image_path = './packages/train_videos/%.3d/disparity/%.8df.raw' % (video_no, frame_no)
annotation_path      = './packages/train_annotations/%.3d.json' % video_no

with open(annotation_path, encoding="UTF-8") as f:
    annotation = json.load(f)
inf_DP = annotation["sequence"][frame_no]['inf_DP']

img1 = get_distance_image(raw_image_path=disparity_image_path, inf_DP=inf_DP)

img2 = load_distance_image(file_path=disparity_image_path, inf_DP=inf_DP)

print(img1 - img2)
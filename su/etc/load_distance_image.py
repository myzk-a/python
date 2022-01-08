import numpy as np

def load_distance_image(file_path, inf_DP):

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

    disparity[disparity <= inf_DP] = 0
    disparity[disparity > inf_DP] = 560.0 / (disparity[disparity > inf_DP] - inf_DP)
    return disparity
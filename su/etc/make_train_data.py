import json
import cv2
import pandas as pd
import numpy as np

def make_train_data(video_path, annotation_path, data_dir, video_no):

    with open(annotation_path, encoding="UTF-8") as f:
        annotation = json.load(f)

    df = pd.DataFrame(annotation["sequence"])
    df_pos = df[["TgtXPos_LeftUp", "TgtYPos_LeftUp", "TgtWidth", "TgtHeight"]].astype("int64") # 先行車の矩形情報を抽出

    # 動画ファイルを読込
    video = cv2.VideoCapture(video_path)

    # 動画情報
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_no in range(frame_count):
        rand = np.random.rand()
        if rand > 0.1:
            save_dir = data_dir + 'train/'
        else:
            save_dir = data_dir + 'valid/'
           
        ret, img = video.read()

        name = str(video_no) + '_' + str(frame_no)
        cv2.imwrite(save_dir+'images/'+name+'.jpg', img)

        x_left = df_pos.iloc[frame_no]["TgtXPos_LeftUp"]
        x_right = x_left + df_pos.iloc[frame_no]["TgtWidth"]
        center_x = 0.5 * (x_left + x_right) / float(width)

        y_low = df_pos.iloc[frame_no]["TgtYPos_LeftUp"]
        y_high = y_low + df_pos.iloc[frame_no]["TgtHeight"]
        center_y = 0.5 * (y_low + y_high) / float(height)
        w        = df_pos.iloc[frame_no]["TgtWidth"] / float(width)
        h        = df_pos.iloc[frame_no]["TgtHeight"] / float(height)
        s = '0' + ' ' + str(center_x) + ' ' + str(center_y) + ' ' + str(w) + ' ' + str(h)
        with open(save_dir+'labels/'+name+'.txt', 'w') as f:
            f.write(s)

    video.release()
    cv2.destroyAllWindows()

for video_no in range(21):
    video_path      = './packages/train_videos/%.3d/Right.mp4' % video_no
    annotation_path = './packages/train_annotations/%.3d.json' % video_no
    data_dir        =  './packages/data_tmp/'

    make_train_data(video_path, annotation_path, data_dir, video_no)
    print("Converted {} Video to train date".format(video_no))
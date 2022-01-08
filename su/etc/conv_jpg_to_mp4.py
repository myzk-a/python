import cv2
import pandas as pd
import json

video_nums = 738
annotation_dir = './train/annotations/'

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

video = cv2.VideoWriter('predict_for_train.mp4', fourcc, 10, (1000, 420))

# アノテーションファイルからデータフレームを作成
for video_no in range(video_nums + 1):
    annotation_path = annotation_dir + '%.3d.json' % video_no
    with open(annotation_path, encoding="UTF-8") as f:
        annotation = json.load(f)

    df = pd.DataFrame(annotation["sequence"])
    df.insert(0, 'FrameNo', df.index)
    max_frame_no = int(df.max()["FrameNo"])

    for frame_no in range(1, max_frame_no + 1):
        file_name = './train_image_added_predicted_result/' + str(video_no) + '_' + str(frame_no) + '.jpg'
        img = cv2.imread(file_name)

        if not img is None:
            video.write(img)
        else:
            print("can not open ", video_no, '_', frame_no)
    
video.release()
print('finish')

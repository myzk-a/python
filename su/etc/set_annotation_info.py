import json
import cv2
import pandas as pd
import matplotlib.pyplot as plt

ws = 'packages/'

# 動画データ読み込み
video_no = 0
video_path = ws + 'train_videos_1/%.3d/Right.mp4' % video_no

video = cv2.VideoCapture(video_path)
ret, img = video.read() # 1フレーム目だけ読み込み
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 画像描画（オリジナル）
plt.figure(figsize = (12,6))
plt.imshow(img)
plt.savefig('original.png')

# annotations読み込み
annotation_path = ws + 'train_annotations/%.3d.json' % video_no

with open(annotation_path, encoding="UTF-8") as f:
    annotation = json.load(f)

df = pd.DataFrame(annotation["sequence"])

# 先行車の矩形情報を抽出
df_pos = df[["TgtXPos_LeftUp", "TgtYPos_LeftUp", "TgtWidth", "TgtHeight"]].astype("int64")
print(df_pos.head())

# 読み込んだ画像にannotation情報を追加

frame_no = 0

# 先行車のバウンディングボックス
pt1 = (df_pos.iloc[frame_no]["TgtXPos_LeftUp"], df_pos.iloc[frame_no]["TgtYPos_LeftUp"])
pt2 = (pt1[0] + df_pos.iloc[frame_no]["TgtWidth"], pt1[1] + df_pos.iloc[frame_no]["TgtHeight"])

img = cv2.rectangle(img, pt1, pt2, (0, 0, 255), thickness=3)

# OwnSpeed
test = "OwnSpd:" + str(df.iloc[frame_no]["OwnSpeed"])
img = cv2.putText(img, test, (10, 40),
              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
              fontScale=1.3,
              color=(255, 0, 0),
              thickness=3)

# StrDeg
test = "StrDeg:" + str(df.iloc[frame_no]["StrDeg"])
img = cv2.putText(img, test, (10, 40 + 50),
              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
              fontScale=1.3,
              color=(255, 0, 0),
              thickness=3)

# Distance_ref
test = "Distance:" + str(df.iloc[frame_no]["Distance_ref"])
img = cv2.putText(img, test, (10, 40 + 100),
              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
              fontScale=1.3,
              color=(255, 0, 0),
              thickness=3)

# TgtSpeed_ref
test = "TgtSpd:" + str(df.iloc[frame_no]["TgtSpeed_ref"])
img = cv2.putText(img, test, (10, 40 + 150),
              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
              fontScale=1.3,
              color=(255, 0, 0),
              thickness=3)

# 画像描画（アノテーション付）
plt.figure(figsize = (12,6))
plt.imshow(img)
plt.savefig('add_annotation.png')
import cv2
import torch
import json
import pandas as pd
import numpy as np

def spd_abs_error(df, pr_col, first_frame=20, limit_gradient=0.07, limit_intercept=3):
    all_error = 0
    weights = 0
    
    for video_no, df_group in df.groupby("VideoNo"):
        df_tmp = df_group.iloc[first_frame-1:]
        limit = limit_gradient*df_tmp["TgtSpeed_ref"] + limit_intercept
        err = np.minimum(np.abs((df_tmp[pr_col] - df_tmp["TgtSpeed_ref"])/limit), 1).mean()
        all_error += df_tmp['Weight'].mean()*err
        weights += df_tmp['Weight'].mean()

    return all_error/weights

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

def set_anotation(df, video_no, frame_no, img, predict_result):
    # シーン番号
    test = str(video_no) + "_" + str(frame_no)
    img = cv2.putText(img, test, (10, 40),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0,color=(0, 0, 255),thickness=3)

    # Distance_ref
    test = "Distance_ref:" + str(df.iloc[frame_no]["Distance_ref"])
    img = cv2.putText(img, test, (10, 40+50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0,color=(0, 0, 255),thickness=3)

    # TgtSpeed_ref
    test = "TgtSpd_ref:" + str(df.iloc[frame_no]["TgtSpeed_ref"])
    img = cv2.putText(img, test, (10, 40+100),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0,color=(0, 0, 255),thickness=3)

    if not predict_result.pandas().xywh[0].empty:
        # 予測したバウンディングボックス
        x_min = int(predict_result.pandas().xyxy[0]['xmin'][0])
        y_min = int(predict_result.pandas().xyxy[0]['ymin'][0])
        x_max = int(predict_result.pandas().xyxy[0]['xmax'][0])
        y_max = int(predict_result.pandas().xyxy[0]['ymax'][0])
        pt1 = (x_min, y_min)
        pt2 = (x_max, y_max)
        img = cv2.rectangle(img, pt1, pt2, (0, 0, 255), thickness=3)

        # 予測した車間距離
        test = "Distance:" + str(df.iloc[frame_no]["Distance"])
        img = cv2.putText(img, test, (10, 40+150),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0,color=(0, 255, 255),thickness=3)

        # 予測した先行車車速
        test = "TargetSpeed:" + str(df.iloc[frame_no]["TargetSpeed"])
        img = cv2.putText(img, test, (10, 40+200),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0,color=(0, 255, 255),thickness=3)

    return img

def predict_preceeding_vehicle_speed(df, video_path, disparity_image_dir, video_no, model, f_save_img=False):
    # 動画ファイルを読込
    video = cv2.VideoCapture(video_path)
    max_frame_no = int(df[df["VideoNo"] == video_no].max()["FrameNo"])
    for frame_no in range(max_frame_no + 1):
        ret, img = video.read()
        cv2.imwrite('tmp.jpg', img)
        image = 'tmp.jpg'
        #image = image_dir + str(video_no) + "_" + str(frame_no) + ".jpg"
        # 先行車の位置を検出
        if(frame_no == 0) :
            x_leftup = df.at[0, "TgtXPos_LeftUp"]
            width = df.at[0, "TgtWidth"]
            x_center = (2*x_leftup + width)/2
            y_leftup = df.at[0, "TgtYPos_LeftUp"]
            height = df.at[0, "TgtHeight"]
            y_center = (2*y_leftup + height)/2
        else:
            results = model(image)
            if not results.pandas().xywh[0].empty:
                x_center = results.pandas().xywh[0]['xcenter'][0]
                y_center = results.pandas().xywh[0]['ycenter'][0]
                    
        print(video_no, '_', frame_no, ' : ', 'x_center = ', x_center, ' y_center = ', y_center)
        # 視差画像から距離を取得
        disparity_image_path = disparity_image_dir + '%.3d/disparity/%.8df.raw' % (video_no, frame_no)
        disparity = load_distance_image(file_path=disparity_image_path, inf_DP=df.at[frame_no, "inf_DP"])
        x_center = int((x_center * 256)/1000)
        y_center = int((y_center * 105)/420)
        distance = disparity[y_center][x_center]
        df.at[frame_no, "Distance"] = distance

        # 先行車の車速を推定
        if(frame_no > 0):
            rel_s = ((distance - df.at[frame_no-1, "Distance"]) / 0.1) * 3.6
            target_speed = df.at[frame_no, "OwnSpeed"] + rel_s
            if(frame_no > 1):
                target_speed = min(target_speed, df.at[frame_no-1, "TargetSpeed"] + 3)
                target_speed = max(target_speed, df.at[frame_no-1, "TargetSpeed"] - 3)
            df.at[frame_no, "TargetSpeed"] = target_speed

        if f_save_img and frame_no > 0:
            img = set_anotation(df=df, video_no=video_no, frame_no=frame_no, img=img, predict_result=results)
            file_name = './train_image_added_predicted_result/' + str(video_no) + "_" + str(frame_no) + ".jpg"
            cv2.imwrite(filename=file_name, img=img)
        
    return df

# 検出器の準備
model = torch.hub.load('./yolov5', 'custom', path='./yolov5/best.pt', source='local')

# オプション設定
model.max_det = 1
model.rect = True

# testデータとtrainデータの切り替え
f_test_data = False

# パス設定
if f_test_data:
    # testデータ使用時
    video_nums = 1
    video_dir = './test_videos/'
    annotation_dir = './test/annotations/'
    image_dir = './test/jpg/'
    disparity_image_dir = './test_videos/'
else:
    # trainデータ使用時
    video_nums = 738
    annotation_dir = './train/annotations/'
    image_dir = './train/jpg/'
    video_dir = './train_videos/'
    disparity_image_dir = './train_videos/'

# アノテーションファイルからデータフレームを作成
df = pd.DataFrame()
for video_no in range(video_nums + 1):
    annotation_path = annotation_dir + '%.3d.json' % video_no
    video_path = video_dir + '%.3d/Right.mp4' %video_no
    with open(annotation_path, encoding="UTF-8") as f:
        annotation = json.load(f)

    df_tmp = pd.DataFrame(annotation["sequence"])
    df_tmp.insert(0, 'VideoNo', video_no)
    df_tmp.insert(1, 'FrameNo', df_tmp.index)
    df_tmp.insert(2, 'Distance', 100.0)
    df_tmp.insert(3, 'TargetSpeed', df_tmp['OwnSpeed'])
    if annotation["attributes"]['評価値計算時の重み付加'] == "有":
        df_tmp.insert(4, 'Weight', 1)
    else:
        df_tmp.insert(4, 'Weight', 0)
    # 先行車の車速を推定
    df_tmp = predict_preceeding_vehicle_speed(df=df_tmp, video_path=video_path, disparity_image_dir=disparity_image_dir, video_no=video_no, model=model, f_save_img=True)
    df = pd.concat([df, df_tmp])

df_test = df.reset_index(drop=True)
print(df_test.head())

# 提出用フィアル作成
target_col = 'TargetSpeed'
if f_test_data:
    json_path = "submit.json"
else:
    json_path = "train_data.json"


predict_dict = {}
for name, df_group in df_test.groupby("VideoNo"):
    predict_dict[str(name).zfill(3)] = df_group[target_col].tolist()
    
with open(json_path, 'w') as f:
    json.dump(predict_dict, f, ensure_ascii=False)

# trainデータ時はスコアを計算
if not f_test_data:
    score = spd_abs_error(df, "TargetSpeed")
    print("スコア：{}".format(score))
    best_score = spd_abs_error(df, "TgtSpeed_ref")
    print("ベストスコア：{}".format(best_score))

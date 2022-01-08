import json
import pandas as pd

df = pd.DataFrame()

for video_no in range(187):
    annotation_path = './packages/train_annotations/%.3d.json' % video_no

    with open(annotation_path, encoding="UTF-8") as f:
        annotation = json.load(f)

    df_tmp = pd.DataFrame(annotation["sequence"])
    df_tmp.insert(0, 'VideoNo', video_no)
    df_tmp.insert(1, 'FrameNo', df_tmp.index)
    if annotation["attributes"]['評価値計算時の重み付加'] == "有":
        df_tmp.insert(2, 'Weight', 1)
    else:
        df_tmp.insert(2, 'Weight', 0)

    df = pd.concat([df, df_tmp])

df_train = df.reset_index(drop=True)

print(df_train.head())

df_train.to_csv("./packages/train_annotations.csv", index=False)
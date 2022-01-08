import cv2
import numpy as np

def conv_mp4_to_jpg(video_path, video_no):
    # 動画ファイルを読込
    video = cv2.VideoCapture(video_path)

    # 動画情報
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_no in range(frame_count):
        ret, img = video.read()
        name = str(video_no) + '_' + str(frame_no)
        cv2.imwrite('./packages/data/test/'+name+'.jpg', img)

    video.release()
    cv2.destroyAllWindows()

for video_no in range(21):
    video_path = './packages/test_video/%.3d/Right.mp4' % video_no
    conv_mp4_to_jpg(video_path=video_path, video_no=video_no)


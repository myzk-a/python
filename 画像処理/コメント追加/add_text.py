#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PIL import Image, ImageDraw, ImageFont


# In[ ]:


text = 'Rui chan!!!'             # 画像に追加する文字列を指定
img = Image.open('rui.jpg') # 入力ファイルを指定

imagesize = img.size        # img.size[0]は幅、img.size[1]は高さを表す
draw = ImageDraw.Draw(img)  # ImageDrawオブジェクトを作成

font = ImageFont.truetype("arial.ttf", 48)  # フォントを指定、64はサイズでピクセル単位
size = font.getsize(text)

#draw.text((imagesize[0] - size[0], imagesize[1] - size[1]), text, font=font, fill='#FFF')
draw.text((0, 0), text, font=font, fill='#FFF')
print(imagesize[0] - size[0])
print(imagesize[1] - size[1])
# ファイルを保存
img.save('rui_text_added.png', 'PNG', quality=100, optimize=True)


# In[ ]:





from PIL import Image


img = Image.open(r'D:\demo.jpg')
img = img.resize((200, 200), Image.ANTIALIAS)
img.save(r'd:\demo1.jpg')
'''
图片转字符画
'''

from PIL import Image
import os
#命令行解析器  https://blog.ixxoo.me/argparse.html
import argparse

parse = argparse.ArgumentParser()
#image参数属于定位参数，也就是必须要填写的
parse.add_argument("--image",help = "指定一个图片路径",type=str,default='./nash.jpg')
#可选参数
parse.add_argument("--width",default=64,type=int,help="图片宽度")
parse.add_argument("--height",type=int,default=64,help="图片高度")
parse.add_argument("-o",'--output',type=str,help="输出结果存放路径")
args = parse.parse_args()

imagePath = args.image
width = args.width
height = args.height
outPath = args.output

ascii_char = list(r'$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^`\'. ')


#rgb转灰度值,并对应到一个字符
def RGB2Gray(r,g,b,alpha=256):
    if alpha == 0:
        return  ' '
    #一个字符对应257中多少值
    length = len(ascii_char)
    unit = (256.0 + 1) / length
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return ascii_char[int(gray/unit)]

if __name__ == "__main__":
    if os.path.exists(imagePath) == False:
        print("图片文件不存在")
    else:
        im = Image.open(imagePath)
        im = im.resize((width, height), Image.NEAREST)

        txt = ''
        #图片像素点转成字符文本
        for i in range(height):
            for j in range(width):
                txt += RGB2Gray(*im.getpixel((j,i)))

            txt += '\n'

    if outPath is None:
        with open('output.txt','w',encoding='utf8') as f:
            f.write(txt)
    else:
        with open(outPath,'w',encoding='utf8') as f:
            f.write(txt)
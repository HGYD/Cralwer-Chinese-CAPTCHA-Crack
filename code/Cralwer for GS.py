#-*- coding:utf-8 -*-
__author__ = 'HGYD'
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import glob
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
import os

# setup location to store data
os.chdir(u"F:\data")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import requests
from requests import Session
try:
    from StringIO import StringIO
    from BytesIO import BytesIO
except ImportError:
    from io import StringIO,BytesIO
from bs4 import BeautifulSoup
from PIL import Image,ImageDraw
from operator import itemgetter
import math
import random
import json


pca = PCA(n_components=0.8,whiten=True)
svc = svm.SVC(kernel='rbf',C=10)
#make sure located these files.
train_opr = pd.read_csv(u"../training_opr.csv")
train_x_opr = train_opr.values[:,1:]
train_y_opr = train_opr.ix[:,0]
#make sure located these files.
train = pd.read_csv(u"../training.csv")
train_x = train.values[:,1:]
train_y = train.ix[:,0]



#get the Captcha
def image_(session):
    url = 'http://gsxt.gdgs.gov.cn/aiccips//verify.html'
    CImages_html=session.get(url,timeout=10)
    file  = BytesIO(CImages_html.content)
    im = Image.open(file)
    data = im.getcolors()
    color_list =[]
    for x in data:
        if  x[0] > 120 and x[0] < 1000 :
            color_list.append(x[1])
    if len(color_list) == 5:
        return im
    else:
        return image_(html)

#get image's color information
def color_(im):
    data = im.getcolors()
    color_list =[]
    for x in data:
        if  x[0] > 120 and x[0] < 1000 :
            color_list.append(x[1])
    return color_list

#clean background
def clear_back(img,data,all_data):
    w, h = img.size
    for y in range(h):
        for x in range(w):
            pixel = img.getpixel((x, y))
            if not pixel in all_data:
                img.putpixel((x, y), (255, 255, 255))
#             elif pixel != data and pixel in all_data:     #如果去掉的噪音上面和下面有相同的点就补上。
# #                 try:
#                     up = y > 0 and img.getpixel((x, y-1)) or None
#                     down = y < h-1 and img.getpixel((x, y+1)) or None
#                     right = x > 0 and img.getpixel((x-1,y)) or None
#                     left = x < w-1 and img.getpixel((x+1,y)) or None
#                     up_down = up or down
#                     right_left = right or left
#                     img.putpixel((x, y), up_down and up_down and right_left or 0)
# #                 except:
# #                     pass
            elif pixel != data:
                img.putpixel((x, y), (255, 255, 255)) # 将表格颜色设置为白色
            else:
                img.putpixel((x, y), (0,0,0))
    for y in range(h):
        for x in range(w):
            pixel2 = img.getpixel((x, y))
            if pixel2 != (0,0,0):
                img.putpixel((x, y), (255, 255, 255)) # 将表格颜色设置为白色
    return img

#two value
def twoValue(image,G):
    t2val = {}
    for y in xrange(0,image.size[1]):
        for x in xrange(0,image.size[0]):
            g = image.getpixel((x,y))
            if g > G:
                t2val[(x,y)] = 1
            else:
                t2val[(x,y)] = 0
    return t2val

# reduce noise
# G: Integer the point that you make a pixel is noise or not
# N: Integer  0 <N <8   1 means considered the point is a noise if there only 1 pixel around the point.
# Z: Integer  number of times to do the work

def clearNoise(t2val,image,N,Z):
    for i in xrange(0,Z):
        t2val[(0,0)] = 1
        t2val[(image.size[0] - 1,image.size[1] - 1)] = 1
        for x in xrange(1,image.size[0] - 1):
            for y in xrange(1,image.size[1] - 1):
                nearDots = 0
                L = t2val[(x,y)]
                if L == t2val[(x - 1,y - 1)]:
                    nearDots += 1
                if L == t2val[(x - 1,y)]:
                    nearDots += 1
                if L == t2val[(x- 1,y + 1)]:
                    nearDots += 1
                if L == t2val[(x,y - 1)]:
                    nearDots += 1
                if L == t2val[(x,y + 1)]:
                    nearDots += 1
                if L == t2val[(x + 1,y - 1)]:
                    nearDots += 1
                if L == t2val[(x + 1,y)]:
                    nearDots += 1
                if L == t2val[(x + 1,y + 1)]:
                    nearDots += 1
                if nearDots < N:
                    t2val[(x,y)] = 1
    return t2val

def saveImage(t2val,filename,size): #redraw
    image = Image.new("1",size)
    draw = ImageDraw.Draw(image)
    for x in xrange(0,size[0]):
        for y in xrange(0,size[1]):
            draw.point((x,y),t2val[(x,y)])
            if x < 10 or y == 0 or x == 179 or y == 39:
                image.putpixel((x, y), (255, 255, 255))
    image.save(filename)
#two value
def final_clear(img):
    w, h = img.size
    for y in range(h):
        for x in range(w):
            pixel = img.getpixel((x, y))
            if pixel > 100:
                img.putpixel((x,y), (255))
            else:
                img.putpixel((x,y), (0))
# #                 except:
# #                     pass
    return img

#get board for each images
def box_(img):
    w, h = img.size
    height = []
    width = []
    for y in range(h):
        for x in range(w):
            pixel = img.getpixel((x,y))
            if pixel == 0:
                height.append(y)
                width.append(x)
    return [min(width),min(height),max(width),max(height)]



def cut_photo(image,image_size,image_rank,i):
    h_center =int( math.ceil((image_size[2]+image_size[0])/2))
    if image_size[-1] > 34:
        v_center = int(math.ceil((image_size[1]+image_size[3])/2))
    else:
        v_center = 17
    left = int(h_center - 17)
    up = int( v_center - 17)
    right = int(h_center +17)
    down =int( v_center + 17)
    new_image = image.crop((left,up,right,down))
    # new_image.show()
    # image_name = raw_input()+'.jpg'
    # number = random.randrange(0,1000000)
    if image_size == image_rank[1]:
        new_image.save('opeator'+str(i)+'.jpg')
    elif image_size == image_rank[0]:
        new_image.save('num1'+str(i)+'.jpg')
    else:
        new_image.save('num2'+str(i)+'.jpg')


def twoValue2(image,G):
    image_arrary = []
    for y in xrange(0,image.size[1]):
        for x in xrange(0,image.size[0]):
            g = image.getpixel((x,y))
            # print g
            if g > G:
                image_arrary.append(0)
            else:
                image_arrary.append(1)
    return image_arrary

def parse(train_x,train_y,test):
    test_x = np.array(test)
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)
    svc.fit(train_x, train_y)
    test_y = svc.predict(test_x)
    return test_y

def pipline(im):
    image = im.convert('1')
    image_arrary=twoValue2(image,100)
    return image_arrary

def answer_to_code(num1,num2,opeator):
    nb1,nb2 = int(num1), int(num2)
    if opeator == '+':
        return  (nb1 + nb2)
    elif opeator == '-':
        return (nb1 - nb2)
    else:
        return  (nb1 * nb2)


def html(key,code,session):
    url = 'http://gsxt.gdgs.gov.cn/aiccips/CheckEntContext/checkCode.html'
    headers = {
        'Host': 'gsxt.gdgs.gov.cn',
        'Connection': 'keep-alive',
        'Origin': 'http://gsxt.gdgs.gov.cn',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.73 Safari/537.36',
        'Referer': 'http://gsxt.gdgs.gov.cn/aiccips/CheckEntContext/showInfo.html'
    }
    data = {
        'textfield':key,
        'code' : str(code)
    }
    req = session.post(url,headers=headers,data=data)
    content = json.loads(req.content)
    return content

def get_link(textfield,code,session):
    url1 = 'http://gsxt.gdgs.gov.cn/aiccips/CheckEntContext/showInfo.html'
    headers1 = {
        'Host': 'gsxt.gdgs.gov.cn',
        'Connection': 'keep-alive',
        'Origin': 'http://gsxt.gdgs.gov.cn',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.73 Safari/537.36',
    }
    data1 = {
        'textfield':textfield,
        'code' : str(code)
    }
    req1 = session.post(url1,headers=headers1,data=data1,timeout=20)
    content1 = req1.content
    link_list = []
    soup = BeautifulSoup(content1,'lxml')
    links = soup.findAll('li' ,class_="font16")
    for link in links:
        a = link.a['href']
        link_list.append(a)
    return link_list

# open your input
with open ('name_list.csv','r') as f:
    b_list = f.readlines()

def packle(session,tag):
    a = image_(session)
    b = color_(a)
    im0 = a.copy()
    im1= a.copy()
    im2= a.copy()
    im3= a.copy()
    im4= a.copy()
    ims={'0':im0,'1':im1,'2':im2,'3':im3,'4':im4}
    rank_list=[]
    for num,im in ims.items():
        im = clear_back(im, b[int(num)],b)
        image =im.convert("L")
        t2val = twoValue(image,100)
        cn = clearNoise(t2val,image,2,8)
        im_name = str(random.randrange(1,1000))+'.jpg'
        saveImage(cn,im_name,image.size)
        image_new = Image.open(im_name)
        image_new = final_clear(image_new)
        ims.update({num:im})
        box = box_(image_new)+[str(num)]
        rank_list.append(box)
    value_image = sorted(rank_list, key=itemgetter(0))[:3]
    image_rank  = [value_image[0][0:4],value_image[1][0:4],value_image[2][0:4]]
    ims_new = {}
    i = random.randrange(1,1000)
    for key in ims.keys():
        for rank in value_image:
            if key in rank:
                ims_new.update({key : ims[key]})
    image_size = {}
    for data in value_image:
        image_size.update({data[-1]:data[:-1]})
    for key in ims_new:
        cut_photo(ims_new[key],image_size[key],image_rank,i)
    num1 = pipline(Image.open('num1'+str(i)+'.jpg'))
    num2 = pipline(Image.open('num2'+str(i)+'.jpg'))
    opeator = pipline(Image.open('opeator'+str(i)+'.jpg'))
    num1_string = parse(train_x,train_y,num1)
    num2_string = parse(train_x,train_y,num2)
    opeator_string = parse(train_x_opr,train_y_opr,opeator)
    answer = answer_to_code(num1_string,num2_string,opeator_string)
    textfield = html(tag,answer,session)
    answer_list=[answer]
    if textfield[u'flag'] == u'1':
        answer_list.append(textfield['textfield'])
        return answer_list
    else: packle(session,tag)



def pool1(b_list):
    tag = b_list.replace('"','').strip()
    session = requests.Session()
    answer_list = packle(session,tag)
    textfield=answer_list[1]
    answer = answer_list[0]
    links = get_link(textfield,answer,session)
    for link in links:
        print link
        link = link+'\n'
        with open ('url_list.txt','a+') as f:
            f.write(link)

    # except:
    #     pass



pool=ThreadPool(2)
pool.map(pool1,b_list)
pool.close()
pool.join()
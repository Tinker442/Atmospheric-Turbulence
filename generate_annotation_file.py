import glob
import pandas as pd
import os
from tqdm import tqdm
import random

def gen_video_annotations(annotations_path = './kinetics400_5per/',source='frames',split=0.8):

    train_path= annotations_path+'train'+'/video_annotations.csv'
    test_path= annotations_path+'test'+'/video_annotations.csv'
    eval_path= annotations_path+'eval'+'/video_annotations.csv'
    vid_paths = glob.glob(annotations_path+'frames'+'/*/*')
    
    total = len(vid_paths)

    train_i = int(total*split)
    test_i = train_i+int(total*(1-split)//2)+1

    sample = random.sample(range(total),total)

    train_vid_paths = [vid_paths[i] for i in sample[:train_i]]
    labels = [vid_path.split('\\')[-2] for vid_path in train_vid_paths]
    df = pd.DataFrame({'vid':train_vid_paths,'label':labels})
    df.to_csv(train_path)

    test_vid_paths = [vid_paths[i] for i in sample[train_i:test_i]]
    labels = [vid_path.split('\\')[-2] for vid_path in test_vid_paths]
    df = pd.DataFrame({'vid':test_vid_paths,'label':labels})
    df.to_csv(test_path)

    eval_vid_paths = [vid_paths[i] for i in sample[test_i:]]    
    labels = [vid_path.split('\\')[-2] for vid_path in eval_vid_paths]
    df = pd.DataFrame({'vid':eval_vid_paths,'label':labels})
    df.to_csv(eval_path)

# def gen_video_annotations(annotations_path = './kinetics400_5per/',source='frames',split=0.8):

#     csv_path= annotations_path+mode+'/video_annotations.csv'
#     img_paths = glob.glob(annotations_path+mode+'/*/frames/*')
#     labels = [image_path.split('\\')[-3] for image_path in img_paths]

#     df = pd.DataFrame({'vid':img_paths,'label':labels})
#     # print(df)
#     df.to_csv(csv_path)
# def gen_image_annotations(annotations_path = './kinetics400_5per/',mode='train'):

#     csv_path= annotations_path+mode+'/image_annotations.csv'
#     img_paths = glob.glob('./kinetics400_5per/train/*/frames/*/*.jpg')
#     labels = [image_path.split('\\')[-4] for image_path in img_paths]

#     df = pd.DataFrame({'img':img_paths,'label':labels})
#     # print(df)
#     df.to_csv(csv_path)

def gen_image_annotations(annotations_path = './kinetics400_5per/',source='frames',split=0.8):

    train_path= annotations_path+'train'+'/image_annotations.csv'
    test_path= annotations_path+'test'+'/image_annotations.csv'
    eval_path= annotations_path+'eval'+'/image_annotations.csv'
    img_paths = glob.glob('./kinetics400_5per/frames/*/*/*.jpg')
    
    total = len(img_paths)
    train_i = int(total*split)
    test_i = train_i+int(total*(1-split)//2)+1
    sample = random.sample(range(total),total)


    train_img_paths = [img_paths[i] for i in sample[:train_i]]
    labels = [img_path.split('\\')[-3] for img_path in train_img_paths]
    df = pd.DataFrame({'img':train_img_paths,'label':labels})
    df.to_csv(train_path)

    test_img_paths = [img_paths[i] for i in sample[train_i:test_i]]
    labels = [img_path.split('\\')[-3] for img_path in test_img_paths]
    df = pd.DataFrame({'img':test_img_paths,'label':labels})
    df.to_csv(test_path)

    eval_img_paths = [img_paths[i] for i in sample[test_i:]]    
    labels = [img_path.split('\\')[-3] for img_path in eval_img_paths]
    df = pd.DataFrame({'vid':eval_img_paths,'label':labels})
    df.to_csv(eval_path)



if __name__=='__main__':

    gen_video_annotations()
    gen_image_annotations()


    # dir = './kinetics400_5per\\'
    # vids = glob.glob(dir+'frames/*/frames/*')

    # for vid in tqdm(vids):
    #     split = vid.split('\\')
    #     del split[-2]
    #     if split[-1]=='frames':
    #         continue
    #     dest = '\\'.join(split)
    #     # print()
    #     # print(vid)
    #     # print(dest)
    #     # break
    #     os.rename(vid,dest)
    #     # os.remove


# if False:
#     annotations_path = './kinetics400_5per/train/video_annotations.csv'
#     # img_paths = glob.glob('./kinetics400_5per/train/*/frames/*/*.jpg')
#     img_paths = glob.glob('./kinetics400_5per/train/*/frames/*')
#     # img_paths= ['./kinetics400_5per/train/test/frames/*/1.jpg','./kinetics400_5per/train/abs/frames/*/2.jpg','./kinetics400_5per/train/abs/frames/*/3.jpg']
#     # print(img_paths)
#     labels = [image_path.split('\\')[-3] for image_path in img_paths]

#     df = pd.DataFrame({'vid':img_paths,'label':labels})
#     # print(df)
#     df.to_csv(annotations_path)
#     # test = pd.read_csv(annotations_path,index_col=0)
#     # print()
#     # print(test['label'].tolist())
#     # print(test.iloc[0,0])

# if False:
#     annotations_path = './kinetics400_5per/train/image_annotations.csv'
#     img_paths = glob.glob('./kinetics400_5per/train/*/frames/*/*.jpg')
#     # img_paths = glob.glob('./kinetics400_5per/train/*/frames/*')
#     # img_paths= ['./kinetics400_5per/train/test/frames/*/1.jpg','./kinetics400_5per/train/abs/frames/*/2.jpg','./kinetics400_5per/train/abs/frames/*/3.jpg']
#     # print(img_paths)
#     labels = [image_path.split('\\')[-4] for image_path in img_paths]

#     df = pd.DataFrame({'img':img_paths,'label':labels})
#     # print(df)
#     df.to_csv(annotations_path)
#     # test = pd.read_csv(annotations_path,index_col=0)
#     # print()
#     # print(test['label'].tolist())
#     # print(test.iloc[0,0])

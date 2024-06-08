import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.animation as animation
def animate_images(img,cmap=None,frame_rate=5,save=False,torch=True,rgb=True):
    #need to run 

    # img = np.random.random((100,64,64,3))
    # cmap=None
    # frame_rate=20
    # save=False

    # plt.rcParams['figure.figsize'] = (5,3)
    # plt.rcParams['figure.dpi'] = 100
    # plt.rcParams['savefig.dpi'] = 100
    # plt.rcParams["animation.html"] = "jshtml"  # for matplotlib 2.1 and above, uses JavaScript

    length = img.shape[0]

    frames = [] # for storing the generated images
    fig = plt.figure()

    img=(img+1)/2
    if torch:
        img=img.numpy()
    

    for i in range(length):
        # frames.append([plt.imshow(img[i], cmap=cm.Greys_r,animated=True)])

        img_buffer = img[i]
        if rgb:
            img_buffer=np.transpose(img_buffer,axes=(1,2,0))
        else:
            img_buffer=img_buffer.transpose()
        frames.append([plt.imshow(img_buffer,cmap=cmap,animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=int(1000/frame_rate), blit=True,
                                    repeat_delay=1000)
    if save:
        ani.save('movie.gif')
    plt.show()
    return ani

if __name__=='__main__':
    #TEST
    
    import torch
    from IPython.display import HTML
    ani = animate_images(torch.rand(100,1,64,64)) 
    HTML(ani.to_jshtml())
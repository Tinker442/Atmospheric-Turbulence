import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import RBF

import time
import cv2

class NoiseModel:
  def __init__(self,size=512):
    '''
    Initialize Synthetic Dataset

    args:
      size -- dimension of sample space
    '''
    self.K=None
    self.size = size

  def gauss_filter(self, filter_size,sigma):
    """
    """
    radius = filter_size//2
    x = np.arange(-radius,radius+1,1)
    k = np.exp(-0.5 * (1 / sigma**2) * x**2)
    matrix= np.outer(k,k)
    matrix /= np.sum(matrix)
    return matrix
  
  def generate_covariance(self, size,depth,max_intensity,sigma): #Unused

    x = np.linspace(-max_intensity, max_intensity, size)
    y = np.linspace(-max_intensity, max_intensity, size)
    z = np.linspace(-max_intensity, max_intensity, depth)
    X, Y,Z = np.meshgrid(x, y,z)
    sample_space = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    kernel = RBF(sigma)
    K= kernel.__call__(sample_space,sample_space)
    self.K=K  
    return K


  def simple_noise(self,image,k_size,sig=1):
    return cv2.GaussianBlur(image,(k_size,k_size),sig)
  
  def variance_noise_convolution(self,image,k_size,noise):
        # Get the dimensions of the image and kernel
    image_height, image_width = image.shape
    # Calculate the output dimensions
    image_padded=np.pad(image,k_size//2)

    windows = np.lib.stride_tricks.sliding_window_view(image_padded, (k_size, k_size))

    filters = np.array([self.gauss_filter(k_size,i) for i in noise.flatten()]).reshape((image_height,image_width,k_size,k_size))

    # Compute convolution with element-wise multiplication and sum
    output = np.sum(windows * filters, axis=(2, 3))

    return output
  

  
  # def discrete_variance_noise_convolution(self,image,k_size=5, noise_steps=8):
  #   image_height, image_width = image.size()[-2:]
  #   steps = torch.linspace(0,1.5,noise_steps)+0.001
  #   noise = torch.tensor(cv2.GaussianBlur(cv2.resize((noise_steps-1) * torch.rand(32, 32).numpy(), dsize=(image_width, image_height)), (5, 5), 2))
  #   indeces=torch.round(noise).int()

  #   filters = torch.stack(
  #     [F.conv2d(image, torch.tensor(cv2.getGaussianKernel(k_size, s.item()).dot(cv2.getGaussianKernel(k_size, s.item()).T)).float().unsqueeze(0).unsqueeze(0),  padding=k_size//2) for s in steps], 
  #     dim=0)
    

  #   output = filters[indeces,:,torch.arange(image_height).unsqueeze(1), torch.arange(image_width).unsqueeze(0)]

  #   return output

  
  def discrete_variance_noise_convolution(self,clips,device,k_size=3, noise_steps=4,max_sigma=4,min_sigma=0.001):
    # print(images)
    batch,sequence_length, channels, image_height, image_width = clips.size()
    steps = torch.linspace(min_sigma,max_sigma,noise_steps).to(device)
    
    noise = torch.stack([
      torch.Tensor(cv2.blur(
          cv2.resize(
              torch.clamp((noise_steps-1) * (2*torch.rand(32, 32)-1) , min = 0).numpy(), dsize=(image_width, image_height)
            ), (5, 5))).repeat(channels,1,1).to(device)
        for i in range(sequence_length)], dim=0)
    
    indeces=torch.floor(noise).to(torch.int64).to(device)

    kernel_fun = lambda s: torch.tensor(cv2.getGaussianKernel(k_size, s.item()).dot(cv2.getGaussianKernel(k_size, s.item()).T),dtype=torch.float32)

    output=[]
    for images in clips:
      # print(images.get_device())

      filters = torch.stack(
        [F.conv2d(images.to(device), kernel_fun(s).repeat(channels,1,1,1).to(device),  padding=k_size//2,groups=3) for s in steps], 
        dim=0)

      # output = filters[indeces,:,:,torch.arange(image_height).unsqueeze(1), torch.arange(image_width).unsqueeze(0)]
      # output = filters[0]
      output.append(filters.gather(0, indeces.unsqueeze(0)).squeeze())
    output=torch.stack(output,dim=0)
    return output
  


  

  



if __name__ == "__main__":
    from animate_images import animate_images
    from IPython.display import HTML
    from torchvision.io import read_image,ImageReadMode
    from torchvision.transforms import Resize




    noise_model=NoiseModel()
    k_size=5
    im_size=128
    image = read_image("./sample_frame.jpg",mode=ImageReadMode.RGB).type(torch.float32)/255
    resize=Resize((im_size,im_size),antialias=True)
    image=resize(image)
  

    clip=[]
    
    for i in range(20):
      
      # noise = cv2.GaussianBlur(cv2.resize(noises[i],dsize=(im_size,im_size)),(5,5),1)
      # output.append( noise_model.variance_noise_convolution(image,k_size,noise))
      clip.append(image)
      # clip.append(torch.Tensor(cv2.GaussianBlur(image.numpy(),(k_size,k_size),2)))
      # output.append(noise_model.simple_noise(image,k_size))
      # 
      # plt.imshow(output,cmap='gray')
      # plt.show()
    
    clip=torch.stack(clip,dim=0).unsqueeze(0)
    # clip=np.dstack(clip).transpose()
    time_start=time.time()
    output=noise_model.discrete_variance_noise_convolution(clip,"cuda:0").cpu()
    # output=clip
    print(time.time()-time_start)
    ani=animate_images(output.squeeze(),torch=True,rgb=True,frame_rate=10)
    HTML(ani.to_jshtml())
    
    # m=noise_model.gauss_filter(5,10)
    
#     size=24
#     depth=10
#     noise_model.generate_covariance(size=size,depth=depth,max_intensity=.5,sigma=.1)
# # 
    
#     sample = noise_model.noise_schedule(size=size,depth=depth)
#     sample = np.maximum(0,sample.T-1)
#     sample = np.array([cv2.resize(frame,dsize=(512,512)) for frame in sample])

#     ani=animate_images(sample,torch=False,rgb=False,frame_rate=15)
#     HTML(ani.to_jshtml())
    
from matplotlib.image import imread
import numpy as np
import os
import matplotlib.pyplot as plt
img= imread("image.JPG" )# importing image as a matrix
X=np.mean(img,-1)# converting into 2d matrix

U,S,VT=np.linalg.svd(X,full_matrices=False)# calculating SVD
# converting Sigma into diagnol matrix


r=[15,50,100]

def image_display(U,S,VT,r):
  figure, axis=plt.subplots(1,3,figsize=(12,4))
  for ax,i in zip(axis,r):# compressing  the image in diffrent rank
    compressed=S[:i]* U[:,0:i]  @ VT[0:i,:]#calculating compresed image
    rotated=np.rot90(compressed,k=3)
    ax.imshow(rotated,cmap="gray")
  
    ax.set_title("r="+str(i))
    ax.axis("off")
  plt.tight_layout()
  plt.show()




def graph(U,S,VT):
  plt.figure(1)
  plt.semilogy(S)
  plt.title("Log of singular value 'S'")
  plt.ylabel("log(S)")
  plt.xlabel("Rank")
  plt.figure(2)
  plt.plot(np.cumsum(S)/np.sum(S))
  plt.ylabel("Energy")
  plt.xlabel("Rank")
  plt.title("Energy distribution")

  plt.show()

graph(U,S,VT)
#image_display(U,S,VT,r)

 


 







import code
import os
import skimage
from skimage import io
from skimage import data, filters, color, morphology, img_as_float, exposure
import matplotlib.pyplot as plt
import numpy as np

def ndim2flat(img):
    np_img = np.array(img)
    
    if len(np.shape(np_img)) == 2:
        flat = img.reshape((np.shape(np_img)[0]*np.shape(np_img)[1],))
    elif len(np.shape(np_img)) == 3:
        flat = img.reshape((np.shape(np_img)[0]*np.shape(np_img)[1]*np.shape(np_img)[2]))        
    return flat

def flatten(img):
  np_img = np.array(img)
  cnt = 0
  y_len = np.shape(np_img)[0];
  x_len = np.shape(np_img)[1];
  flat_arr = np.zeros((y_len*x_len))
  
  # Create 1 dimensional array
  for yy in range(y_len-1):
      for xx in range(x_len-1):
          flat_arr[cnt] = np_img[yy,xx]        
          cnt += 1
  return flat_arr

def thresh_uniform(img, num_bins):
  new_img = np.array(img)
  mmax = np.amax(img)
  mmin = np.amin(img)
  sspan = mmax-mmin
  seg = sspan/num_bins
  half = seg/2
  
  for ii in range(num_bins):
      new_img = thresh(new_img, half, half, (ii*seg)+half, (ii*seg)+half)
      
  return new_img

def thresh_uniform_ends(img, num_bins):
  new_img = np.array(img)
  mmax = np.amax(img)
  mmin = np.amin(img)
  sspan = mmax-mmin
  seg = sspan/num_bins
  half = seg/2
  
  for ii in range(num_bins):
      if ii == 0:
        new_img = thresh(new_img, half, 0, mmin, mmin)
        continue
      if ii == num_bins-1:
        new_img = thresh(new_img, 0, half, mmax, mmax)
        continue
      
      new_img = thresh(new_img, half, half, (ii*seg)+half, (ii*seg)+half)
      
  return new_img

def thresh(img, upper, lower, center, val):
  cnt = 0
  new_img = np.array(img)
  y_len = np.shape(img)[0];
  x_len = np.shape(img)[1];
  
  for yy in range(y_len-1):
      for xx in range(x_len-1):
          if img[yy,xx] > center - lower and img[yy,xx] <= center + upper:
            new_img[yy,xx] = val;
        
  return new_img

def comp_eq(img, bins, name):
    img = img_as_float(img)
    plt.subplot(221)    
    plt.imshow(img, cmap='gray')
    plt.subplot(222)
    img_hist = flat2hist(flatten(img.astype(int)), bins, 0)
    plt.bar(img_hist[0], img_hist[1], color = 'blue')
    plt.xlabel('Bin')
    plt.ylabel('Count')
    plt.title('Histogram')
    img_eq = exposure.equalize_hist(img)
    plt.subplot(223)    
    plt.imshow(img_eq, cmap='gray')
    plt.subplot(224)
    img_hist = flat2hist(flatten(img_eq.astype(int)), bins, 0)
    plt.bar(img_hist[0], img_hist[1], color = 'blue')
    plt.xlabel('Bin')
    plt.ylabel('Count')
    plt.title('Equalized Histogram')
    plt.show()


def thresh_div(img, val):
  cnt = 0
  new_img = np.array(img)
  y_len = np.shape(img)[0];
  x_len = np.shape(img)[1];
  mmax = np.amax(img)
  mmin = np.amin(img)
  img_span = mmax-mmin
  mmid = mmin + (img_span/2)
  
  for yy in range(y_len-1):
      for xx in range(x_len-1):
          if img[yy,xx] < mmid:
            new_img[yy,xx] = val
  return new_img            


def thresh_div(img, thresh_val, new_val):
  cnt = 0
  new_img = np.array(img)
  y_len = np.shape(img)[0];
  x_len = np.shape(img)[1];
  mmax = np.amax(img)
  mmin = np.amin(img)
  img_span = mmax-mmin
  mmid = mmin + (img_span/2)
  
  for yy in range(y_len-1):
      for xx in range(x_len-1):
          if img[yy,xx] < thresh_val:
            new_img[yy,xx] = new_val
  return new_img            


def flat2hist(flat_img, num_bins, plot_enable):

  if plot_enable == 1:
      print('Plot enabled')
  
  img = np.array(flat_img)
  img_max = np.amax(img)
  img_min = np.amin(img)
  img_len = np.shape(img)[0]
  epsilon = float((img_max - img_min)/num_bins)
  bins = np.arange(img_min, img_max, epsilon)
  bins_count = np.zeros((len(bins)))
  bins_len = len(bins)

  # Create 256 bins for histogram        
  for ii in range(img_len-1):
      for bin in range(bins_len-1):
          if img[ii] >= bins[bin] and img[ii] < bins[bin] + epsilon:
              bins_count[bin] += 1
              break
  return np.array([bins, bins_count])

def color2grey(img):
    b = [.3, .6, .1]
    return np.dot(img[...,:3], b)

def comp_label_gray_debug(img, x, y, targ_val, tolerance, nbrhd_width):
    count = 0
    half = int(nbrhd_width/2)
    new_img = np.array(img[y-half:y+half+1, x-half:x+half+1])
    plt.imshow(new_img, cmap='gray')
    plt.show()


def majority(nbrhd, targ_val, tolerance):
    y_len = np.shape(nbrhd)[0];
    x_len = np.shape(nbrhd)[1];
    sum1 = 0
    count = 0
    is_majority = 0
    for yy in range(y_len):
        for xx in range(x_len):
            if (nbrhd[yy, xx] >= targ_val - tolerance) and (nbrhd[yy, xx] < targ_val + tolerance):
              sum1+= 1
            count+=1

    if ((sum1/count) >= 0.5):
        is_majority = 1
        
    return is_majority

def comp_label_gray(img, y, x, targ_val, tolerance, nbrhd_width):
    count = 0
    half = int(nbrhd_width/2)
    y_len = np.shape(img)[0];
    x_len = np.shape(img)[1];
    x_neg = x-half
    y_neg = y-half
    y_over = y_len-(y+half)-1
    x_over = x_len-(x+half)-1

    ## Top left --
    if (x_neg < 0) and (y_neg < 0):
      new_img = np.array(img[0:y+half+1, 0:x+half+1])
    ## Top right --
    elif (x_over < 0) and (y_neg < 0):
      new_img = np.array(img[0:y+half+1, x-half:x_len])                
    ## bottom left --
    elif (x_neg < 0) and (y_over < 0):
      new_img = np.array(img[y-half:y_len, 0:x+half+1])        
    ## bottom right
    elif (x_over < 0) and (y_over < 0):
      new_img = np.array(img[y-half:y_len, x-half:x_len]) 
    ## Top --
    elif (y_neg < 0):
      new_img = np.array(img[0:y+half+1, x-half:x+half+1])
    ## Bottom --
    elif (y_over < 0):
      new_img = np.array(img[y-half:y_len, x-half:x+half+1])        
    ## left --
    elif (x_neg < 0):
      new_img = np.array(img[y-half:y+half+1, 0:x+half+1])        
    ## right --
    elif (x_over < 0):
      new_img = np.array(img[y-half:y+half+1, x-half:x_len])
    else:
      new_img = np.array(img[y-half:y+half+1, x-half:x+half+1])

    #print('comp label gray: ', np.shape(new_img))
    #plt.imshow(new_img, cmap='gray')
    #plt.show()

    return majority(new_img, targ_val, tolerance)

    
def c_label(img, targ_val,tolerance, nbrhd_width):

    if(nbrhd_width % 2 == 0):
        print('nbrhd_width: ', nbrhd_width, '  must be an odd number')
        return 0

    new_img = np.array(img)
    y_len = np.shape(img)[0];
    x_len = np.shape(img)[1];
    for yy in range(y_len):
        for xx in range(x_len):
            if(comp_label_gray(img, yy, xx, targ_val, tolerance, nbrhd_width) > 0):
                new_img[yy, xx] = targ_val;
            #plt.imshow(new_img, cmap='gray')
            #plt.show()
            
    return new_img

def c_label_bins(img, targ_vals,tolerance, nbrhd_width):

    if(nbrhd_width % 2 == 0):
        print('nbrhd_width: ', nbrhd_width, '  must be an odd number')
        return 0

    targ_vals_len = len(targ_vals)
    new_img = np.array(img)
    y_len = np.shape(img)[0];
    x_len = np.shape(img)[1];
    for yy in range(y_len):
        for xx in range(x_len):
            for targs in range(targ_vals_len-1):
              if(comp_label_gray(img, yy, xx, targ_vals[targs], tolerance, nbrhd_width) > 0):
                new_img[yy, xx] = targ_vals[targs];
            #plt.imshow(new_img, cmap='gray')
            #plt.show()
            
    return new_img

def smooth(img, n_pix):

    y_len = np.shape(img)[0];
    x_len = np.shape(img)[1];
    new_img = np.array(img)

    for yy in range(y_len-n_pix):
        for xx in range(x_len-n_pix):
            new_img[yy, xx] = np.sum(img[yy:yy+n_pix,xx:xx+n_pix])/(n_pix*n_pix)

    return new_img

#https://programtalk.com/python-examples/skimage.morphology.remove_small_holes/            
def test_labeled_image_holes():
    labeled_holes_image = np.array([[0,0,0,0,0,0,1,0,0,0],
                                    [0,1,1,1,1,1,0,0,0,0],
                                    [0,1,0,0,1,1,0,0,0,0],
                                    [0,1,1,1,0,1,0,0,0,0],
                                    [0,1,1,1,1,1,0,0,0,0],
                                    [0,0,0,0,0,0,0,2,2,2],
                                    [0,0,0,0,0,0,0,2,0,2],
                                    [0,0,0,0,0,0,0,2,2,2]], dtype=int)

    plt.subplot(221)
    plt.imshow(labeled_holes_image, cmap='gray')
    plt.title('labeled holes')
    
    expected = np.array([[0,0,0,0,0,0,1,0,0,0],
                         [0,1,1,1,1,1,0,0,0,0],
                         [0,1,1,1,1,1,0,0,0,0],
                         [0,1,1,1,1,1,0,0,0,0],
                         [0,1,1,1,1,1,0,0,0,0],
                         [0,0,0,0,0,0,0,1,1,1],
                         [0,0,0,0,0,0,0,1,1,1],
                         [0,0,0,0,0,0,0,1,1,1]], dtype=bool)

    plt.subplot(222)
    plt.imshow(expected, cmap='gray')
    plt.title('expected holes')

    
    observed = morphology.remove_small_holes(labeled_holes_image, 3)
    
    plt.subplot(223)    
    plt.imshow(observed, cmap='gray')
    plt.title('observed holes')
    plt.show()
    
    #assert_array_equal(observed, expected)
    


  





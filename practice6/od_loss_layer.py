from caffe import Layer
import numpy as np

def iou(x, y):
  Ax = (x[:,0] + x[:,1]) * (x[:,2] + x[:,3])
  Ay = (y[:,0] + y[:,1]) * (y[:,2] + y[:,3])
  I_w = np.minimum(x[:,0], y[:,0]) + \
        np.minimum(x[:,1], y[:,1])
  I_h = np.minimum(x[:,2], y[:,2]) + \
        np.minimum(x[:,3], y[:,3])
  AI = I_w * I_h
  AU = Ax + Ay - AI + 1e-6
  return (AI, AU, I_h, I_w)

class ODLossLayer(Layer):
  def setup(self, bottom, top):
    # bottom[0]: predicted box at each pixel
    #            x = (xl, xr, xt, xb)
    # bottom[1]: regression target at each pixel
    #            y = (yl, yr, yt, yb, bg)
    #            bg: 1 if pixel is in background
    #            (pixel of bg = 1 is ignored)
    # bottom[2]: predicted score at each pixel
    #            sx: IoU_x = sigmoid(sx)
    # bottom[3]: true IoU at each pixel, iou_y
    # top[0]: loss_box = ln IoU(x, y)
    # top[1]: loss_score = cross entropy
    top[0].reshape(1)
    top[1].reshape(1)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
    # loss_box
    Y = bottom[1].data
    X = np.rollaxis(bottom[0].data, 1, 4)
    batch_size, h, w, _ = X.shape
    num_p = h * w
    X = X.reshape(batch_size, num_p, 4)
    loss = 1.0
    for x, y in zip(X, Y):
      is_fg = y[:,4] != 1
      x = x[is_fg, :]
      y = y[is_fg, :]
      if len(x) > 0:
        AI, AU, _, _ = iou(x, y)
        #loss -= np.log(AI / AU).sum()
        loss -= (AI / AU).mean() / batch_size
    top[0].data[...] = loss
    # loss_score
    sX = bottom[2].data
    iou_Y = bottom[3].data
    loss = 0.0
    for sx, iou_y in zip(sX, iou_Y):
      loss += (np.log(1 + np.exp(np.minimum(20, sx))) \
               - iou_y * sx).mean() / batch_size
    top[1].data[...] = loss

  def backward(self, top, propagate_down, bottom):
    # d(loss_box) / d(x)
    Y = bottom[1].data
    X = np.rollaxis(bottom[0].data, 1, 4)
    batch_size, h, w, _ = X.shape
    num_p = h * w
    X = X.reshape(batch_size, num_p, 4)
    diff = np.zeros((batch_size, num_p, 4), \
                    dtype=np.float32)
    for n, (x, y) in enumerate(zip(X, Y)):
      is_fg = y[:,4] != 1
      x = x[is_fg, :]
      y = y[is_fg, :]
      AI, AU, I_h, I_w = iou(x, y)
      dAI = np.zeros((is_fg.sum(), 4), \
                     dtype=np.float32)
      dAx = np.zeros((is_fg.sum(), 4), \
                     dtype=np.float32)
      dAI[:,0] = (x[:,0] < y[:,0]) * I_h * \
            (AI + AU) / (AI * AU + 1e-6)
      dAx[:,0] = (x[:,2] + x[:,3]) / AU
      dAI[:,1] = (x[:,1] < y[:,1]) * I_h * \
            (AI + AU) / (AI * AU + 1e-6)
      dAx[:,1] = (x[:,2] + x[:,3]) / AU
      dAI[:,2] = (x[:,2] < y[:,2]) * I_w * \
            (AI + AU) / (AI * AU + 1e-6)
      dAx[:,2] = (x[:,0] + x[:,1]) / AU
      dAI[:,3] = (x[:,3] < y[:,3]) * I_w * \
            (AI + AU) / (AI * AU + 1e-6)
      dAx[:,3] = (x[:,0] + x[:,1]) / AU
      diff[n, is_fg, :] = -dAI + dAx
    diff = np.rollaxis(diff, 2, 1)
    diff = diff.reshape(batch_size, 4, h, w)
    bottom[0].diff[...] = diff
    # d(loss_score) / d(sx)
    sX = bottom[2].data
    iou_Y = bottom[3].data
    for n, (sx, iou_y) in enumerate(zip(sX, iou_Y)):
      diff = 1 / (1 + np.exp(np.minimum(20, -sx))) \
             - iou_y
      bottom[2].diff[n,...] = diff

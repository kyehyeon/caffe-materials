from caffe import Layer

class ODPoolLayer(Layer):
  def setup(self, bottom, top):
    # bottom[0]: predicted score at each pixel
    #   batch_size x num_classes x h x w
    # bottom[1]: predicted box at each pixel
    #   num_boxes x 5
    #   x = (n, p1_j, p1_i, p2_j, p2_i)
    #   n: batch item index (0,...,batch_size-1)
    # top[0]: average score for each box
    #   num_boxes x num_classes

  def reshape(self, bottom, top):
    batch_size, num_classes, h, w = bottom[0].data.shape
    num_boxes = bottom[1].data.shape[1]
    top[0].reshape(batch_size, num_classes, num_boxes)

  def forward(self, bottom, top):
    score = bottom[0].data
    BX = bottom[1].data
    batch_size, num_classes, h, w = score.shape
    num_boxes = BX.shape[1]
    pooled_score = np.zeros(num_boxes, num_classes)
    count = 0
    for n, score in enumerate(score):
      box_n = np.where(BX[:,0] == n)[0]
      Bx = int(round(BX[box_n, :]))
      for bx in Bx:
        pooled_score[count] = \
          score[n, :, Bx[1]:Bx[3], Bx[0]:Bx[2]] \
              .reshape(num_classes, -1) \
              .mean(axis=1)
        count += 1
    top[0].data[...] = pooled_score

  def backward(self, top, propagate_down, bottom):
    pass

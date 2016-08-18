import caffe
import matplotlib.pyplot
import time as timelib
import numpy as np
import sys

# Initialize solver
caffe.set_mode_gpu()
solver = caffe.SGDSolver(sys.argv[1])
solver.net.copy_from(sys.argv[2])

# Initialize figure
fig, axes = matplotlib.pyplot.subplots()
fig.show()

loss_list = []
loss_tags = solver.net.outputs
sub_loss_list = { tag: [] for tag in loss_tags }
max_iter = 121000

window = [0, 1000]
base_lr0 = solver.get_base_lr()
iter0 = solver.iter

while solver.iter < max_iter:
  start = timelib.time()
  solver.step(1)
  time = timelib.time() - start
  loss = [solver.net.blobs[tag].data.flatten() \
          for tag in loss_tags]
  loss = np.array(loss)

  if len(loss_list) == 0:
    mean_loss = loss
    mean_time = time
  else:
    mean_loss = 0.999 * mean_loss + 0.001 * loss
    mean_time = 0.999 * mean_time + 0.001 * time
  total_loss = mean_loss[0]
  loss_list.append(total_loss)
  for i, tag in enumerate(loss_tags):
    sub_loss_list[tag].append(mean_loss[i])

  if len(loss_list) - window[0] > window[1] and \
       total_loss > 0.99 * loss_list[-window[1]]:
    print 'half'
    solver.set_base_lr(solver.get_base_lr() * 0.5)
    window[0] = len(loss_list)
    window[1] *= 2

  if solver.get_base_lr() < 0.1 * base_lr0:
    print 'restore'
    solver.set_base_lr(base_lr0)
    window[1] = 1000
    solver.snapshot()
    fig.savefig('fig_iter_%d.png' % solver.iter)

  # Update plot every 500 iterations
  if solver.iter % 500 == 0:
    axes.clear()
    axes.set_title('Running time = %.3fs/iteration' % mean_time)
    for tag in loss_tags:
      axes.plot(range(iter0, iter0+len(loss_list)), \
                sub_loss_list[tag], label=tag)
    axes.grid(True)
    axes.legend(loc='right')
    fig.canvas.draw()
    matplotlib.pyplot.pause(0.01)

solver.snapshot()
fig.savefig('fig_iter_%d.png' % solver.iter)

import caffe
import matplotlib.pyplot
import time as timelib

caffe.set_mode_gpu()

# Initialize solver
solver = caffe.SGDSolver('examples/cifar10/cifar10_quick_solver.prototxt')

# Initialize figure
fig, axes = matplotlib.pyplot.subplots()
fig.show()

loss_list = []
test_loss_list = []
max_iter = 12100
iter0 = solver.iter

window = [0, 200]
base_lr0 = solver.get_base_lr()

while solver.iter < max_iter:
  start = timelib.time()
  solver.step(1)
  time = timelib.time() - start

  loss = solver.net.blobs['loss'].data.flatten()
  if len(loss_list) == 0:
    mean_loss = loss
    mean_time = time
  else:
    mean_loss = 0.999 * mean_loss + 0.001 * loss
    mean_time = 0.999 * mean_time + 0.001 * time
  loss_list.append(mean_loss)

  if len(loss_list) - window[0] > window[1] and \
       mean_loss > 0.7 * loss_list[-window[1]]:
    print 'half'
    solver.set_base_lr(solver.get_base_lr() * 0.5)
    window[0] = len(loss_list)
    window[1] *= 2

  if solver.get_base_lr() < 0.1 * base_lr0:
    print 'restore'
    solver.set_base_lr(base_lr0)
    window[1] = 200
    solver.snapshot()
    fig.savefig('fig_iter_%d.png' % solver.iter)

  # Update plot every 100 iterations
  if solver.iter % 500 == 0:
    test_loss = solver.test_nets[0].blobs['loss'].data.flatten()
    test_loss_list.append(test_loss)

    axes.clear()
    axes.set_title('Running time = %.3fs/iteration' % mean_time)
    axes.plot(range(iter0, iter0+len(loss_list)), loss_list, label='train_loss')
    axes.plot(range(iter0, iter0+len(loss_list), 500), test_loss_list, label='test_loss')
    axes.grid(True)
    axes.legend(loc='upper right')
    fig.canvas.draw()
    matplotlib.pyplot.pause(0.01)

solver.snapshot()
fig.savefig('fig_iter_%d.png' % solver.iter)

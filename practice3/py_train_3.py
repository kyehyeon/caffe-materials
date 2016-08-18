import caffe
import matplotlib.pyplot
import time as timelib

solver = caffe.SGDSolver('caffe-materials/practice1/solver.pt')

fig, axes = matplotlib.pyplot.subplots()
fig.show()

loss_list = []
max_iter = 10000
iter0 = solver.iter

window = [0, 1000]
base_lr0 = solver.get_base_lr()

while solver.iter < max_iter:
  solver.step(1)

  loss = solver.net.blobs['loss'].data.flatten()
  if len(loss_list) == 0:
    mean_loss = loss
  else:
    mean_loss = 0.99 * mean_loss + 0.01 * loss
  loss_list.append(mean_loss)

  if len(loss_list) - window[0] > window[1] and \
       mean_loss > 0.99 * loss_list[-window[1]]:
    solver.set_base_lr(solver.get_base_lr() * 0.5)
    window[0] = len(loss_list)
    window[1] *= 2

  # Update plot for every 20 iterations
  if solver.iter % 20 == 0:
    axes.clear()
    axes.plot(range(iter0, iter0+len(loss_list)), loss_list)
    axes.grid(True)
    fig.canvas.draw()
    matplotlib.pyplot.pause(0.01)

solver.snapshot()
fig.savefig('fig_iter_%d.png' % solver.iter)

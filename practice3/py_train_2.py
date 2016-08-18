import caffe
import matplotlib.pyplot
import time as timelib

solver = caffe.SGDSolver('caffe-materials/practice1/solver.pt')

fig, axes = matplotlib.pyplot.subplots()
fig.show()

loss_list = []
max_iter = 10000
iter0 = solver.iter

while solver.iter < max_iter:
  solver.step(1)

  loss = solver.net.blobs['loss'].data.flatten()
  loss_list.append(loss)

  # Update plot for every 20 iterations
  if solver.iter % 20 == 0:
    axes.clear()
    axes.plot(range(iter0, iter0+len(loss_list)), loss_list)
    axes.grid(True)
    fig.canvas.draw()
    matplotlib.pyplot.pause(0.01)

solver.snapshot()
fig.savefig('fig_iter_%d.png' % solver.iter)

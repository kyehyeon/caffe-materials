import caffe

# Use GPU 0
#caffe.set_mode_gpu()
#caffe.set_device(0)

# Initialize solver
solver = caffe.SGDSolver('caffe-materials/practice2/solver-cpu.pt')

# Restore snapshot
#solver.restore('')
# or trained parameters
solver.net.copy_from('caffe-materials/practice2/squeezenet_v1.1.caffemodel')

# Train 10 iterations
#solver.step(10)
# Or run 1 forward operation
solver.net.forward()

# Access 'pool10' data (NumPy array)
pool10_data = solver.net.blobs['pool10'].data
print pool10_data.shape   # (32, 1000, 1, 1)

# Save snapshot
solver.snapshot()

import lmdb, cv2, caffe
import numpy as np

def write_lmdb(db_path, list_filename, height, width):
  map_size = 9999999999
  db = lmdb.open(db_path, map_size=map_size)
  writer = db.begin(write=True)
  datum = caffe.proto.caffe_pb2.Datum()
  for index, line in enumerate(open(list_filename, 'r')):
    img_filename, label = line.strip().split(' ')
    img = cv2.imread(img_filename, 1)
    img = cv2.resize(img, (height, width))
    _, img_jpg = cv2.imencode('.jpg', img)
    datum.channels = 3
    datum.height = height
    datum.width = width
    datum.label = int(label)
    datum.encoded = True
    datum.data = img_jpg.tostring()
    datum_byte = datum.SerializeToString()
    index_byte = '%010d' % index
    writer.put(index_byte, datum_byte, append=True)
  writer.commit()
  db.close()

def read_lmdb(db_path):
  db = lmdb.open(db_path, readonly=True)
  reader = db.begin()
  cursor = reader.cursor()
  datum = caffe.proto.caffe_pb2.Datum()
  for index_byte, datum_byte in cursor:
    datum.ParseFromString(datum_byte)
    np_array = np.fromstring(datum.data, dtype=np.uint8)
    label = datum.label
    img = cv2.imdecode(np_array, 1)
    data = np.rollaxis(img, 2, 0)
    yield (data, label)
  cursor.close()
  db.close()

if __name__ == '__main__':
  import sys
  if len(sys.argv) > 2 and sys.argv[1] == 'read':
    for n, (data, label) in enumerate(read_lmdb(sys.argv[2])):
      print 'Sample %d: label=%d, data=%s' % (n, label, data)
  elif len(sys.argv) > 5 and sys.argv[1] == 'write':
    write_lmdb(sys.argv[5], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
  else:
    print 'Usage:'
    print '  lmdb_access read <lmdb_path>'
    print '  lmdb_access write <file_list_txt> <height> <width> <lmdb_path>'

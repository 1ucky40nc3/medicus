import scipy.ndimage
import matplotlib.pyplot as plt

from stl import mesh
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import openmesh as om

from PIL import Image


def make_mesh(image, threshold=-300, step_size=4):

    print("Transposing surface")
    p = image.transpose(2,1,0)
    
    print("Calculating surface")
    if(threshold == -300):
      verts, faces, norm, val = measure.marching_cubes(p, step_size=step_size, allow_degenerate=True) 
    else:  
      verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True) 
    return verts, faces
  
  
  def to_mesh(verts, faces):
    mesh = om.TriMesh()
    vert_list = []
    face_list = []
    for vert in verts:
      vert_list.append(mesh.add_vertex(vert))
    for face in faces:
      face_list.append(mesh.add_face(vert_list[face[0]],vert_list[face[1]],vert_list[face[2]]))


    return mesh
  
  def smoothing(voxel_array, power):
    new_np = []
    factor = 2

    for slice_array in voxel_array:
      slice_img = []
      for row in slice_array:
        row_img = []
        for pix in row:
          for n in range(0,factor):
            row_img.append(pix)
        for n in range(0,factor):
          slice_img.append(row_img)
      for n in range(0,factor):
        new_np.append(slice_img)
    new_np = np.array(new_np)
    new_blur = scipy.ndimage.filters.gaussian_filter(new_np, sigma = 2, truncate=4)

    return new_blur

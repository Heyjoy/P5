import cv2
import numpy as np
from datafield import *

def warp(img):
    img_size=(img.shape[1],img.shape[0])
    warped = cv2.warpPerspective(img,df.M,img_size,flags=cv2.INTER_LINEAR)
    return warped

def unwarp(color_warp,OrgImage):
    unwarped = cv2.warpPerspective(color_warp, df.Minv, (OrgImage.shape[1], OrgImage.shape[0]))
    return unwarped

def makeGridOnWarp():
    xsize, ysize = self.imgsize
    self._xsteps = int(xsize / self.sep[0]) + 1
    self._ysteps = int(ysize / self.sep[1])
    grid = np.mgrid[0:self._xsteps, 0:self._ysteps]
    self.grid_points = grid.T.reshape((-1, 2)) * self.sep
    # Transform the grid points in a perspective view.
    points_transformed = []
    for point in self.grid_points:
        coord = np.append(point, 1)
        transformed = np.dot(self.wp.M_inv, coord)
        point_transformed = (transformed[:2] / transformed[2]).astype(np.int)
        points_transformed.append(point_transformed)
    self.points_transformed = np.array(points_transformed)

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=3):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img,bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img

def find_matches(img, template_list):
    # Define an empty list to take bbox coords
    bbox_list = []
    # Define matching method
    # Other options include: cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCORR',
    #         'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
    method = cv2.TM_CCOEFF_NORMED
    # Iterate through template list
    for temp in template_list:
        # Read in templates one by one
        tmp = mpimg.imread(temp)
        # Use cv2.matchTemplate() to search the image
        result = cv2.matchTemplate(img, tmp, method)
        # Use cv2.minMaxLoc() to extract the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # Determine a bounding box for the match
        w, h = (tmp.shape[1], tmp.shape[0])
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # Append bbox position to list
        bbox_list.append((top_left, bottom_right))
        # Return the list of bounding boxes

    return bbox_list

def plot3d(pixels, colors_rgb,
        axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation


def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict





# plot method
def twoImagePlot(image1,image2,title1 = 'Image1', title2 = 'Image2', color1='gray',color2 ='gray',path = 'output_images/default.png'):
    plt.figure(1)
    plt.axis('off')
    plt.subplot(121)
    plt.title(title1)
    plt.imshow(image1,cmap = color1)
    plt.subplot(122)
    plt.title(title2)
    plt.imshow(image2,cmap = color2 )
    plt.savefig(path)
    plt.show()

def threeImagePlot(image1,image2,image3,color='gray'):
    f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(16, 6))
    f.tight_layout()
    ax1.imshow(image1,cmap=color)
    ax1.set_title('image1', fontsize=50)
    ax2.imshow(image2,cmap=color)
    ax2.set_title('image2', fontsize=50)
    ax3.imshow(image3,cmap=color)
    ax3.set_title('image3', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
class WarpPerspective(object):

    def __init__(self, src=None, dst=None, img_size=(1280, 720)):
        self.img_size = img_size
        if src is None or dst is None:
            xtr,ytr = 690,450   # x,y topRight
            xbr,ybr = 1112,719  # x,y bottomRight
            xbl,ybl = 223,719   # x,y bottomLeft
            xtl,ytl = 596,450   # x,y topLeft

            xtr_dst,ytr_dst = 960,0  # x,y topRight Destination
            xbr_dst,ybr_dst = 960,720  # x,y bottomRight Destination
            xbl_dst,ybl_dst = 320,720   # x,y bottomLeft Destination
            xtl_dst,ytl_dst = 320,0   # x,y topLeft Destination

            self.src = np.float32(
                [[xtr,ytr],
                 [xbr,ybr],
                 [xbl,ybl],
                 [xtl,ytl]])
            self.dst = np.float32(
                [[xtr_dst,ytr_dst],
                 [xbr_dst,ybr_dst],
                 [xbl_dst,ybl_dst],
                 [xtl_dst,ytl_dst]])
        else:
            self.src = np.float32(src)
            self.dst = np.float32(dst)
        # Calculate transform matrix and inverse matrix
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def warp(self, img):
        '''Warp an image from perspective view to bird view.'''
        assert (img.shape[1], img.shape[0]) == self.img_size, 'Invalid image shape.'
        return cv2.warpPerspective(img, self.M, self.img_size)

    def warp_inv(self, img):
        '''Warp inversely an image from bird view to perspective view.'''
        assert (img.shape[1], img.shape[0]) == self.img_size, 'Invalid image shape.'
        return cv2.warpPerspective(img, self.M_inv, self.img_size)


class SlidingWindows(object):

    def __init__(self, wp, imgsize=(1280, 720), sep=[25, 25]):
        self.wp = wp
        self.imgsize = imgsize
        self.sep = sep  # grid point separation
        self._make_grid_points()
        self._make_sliding_windows()

    def _make_grid_points(self):
        '''Generate grid points in a bird view.'''
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

    def _select_ysteps(self, ysteps):
        yseq = np.arange(ysteps - 1)
        yseq_r = yseq[::-1]  # np.flipud(yseq)
        n = np.int(np.sqrt(4 * ysteps + 0.25) + 0.5)
        idx_list = []
        for i in range(n-1):
            tmp = np.int(i * (i+1) / 4)
            idx_list.append(tmp)
        idx_list = np.unique(idx_list)
        return np.unique(np.append(yseq_r[idx_list], 0)[::-1])

    def _make_sliding_windows(self):
        window_list = []
        xseq = range(self._xsteps)
        yseq = self._select_ysteps(self._ysteps)
        for yc in yseq:
            xl = self.points_transformed[yc*self._xsteps][0]
            xr = self.points_transformed[(yc+1)*self._xsteps-1][0]
            width = int( (50/self.sep[0]) * (xr-xl) / (self._xsteps-1) )
            for xc in xseq:
                pos = yc*self._xsteps + xc
                point = self.points_transformed[pos]
                lu = point - np.array([0.5*width, 1.5*width], np.int).tolist()
                rd = point + np.array([1.5*width, 0.5*width], np.int).tolist()
                if 0 < lu[0] < self.imgsize[0] and rd[1] < self.imgsize[1]:
                    window_list.append((tuple(lu), tuple(rd)))
        self.windows = window_list

# store

cars = []
notcars = []

color_space = 'YCrCb'  # RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_size = (8, 8)
hist_bins = 16  # Number of histogram bins
orient = 8  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # 0, 1, 2, or "ALL"
spatial_feat = True
hist_feat = False
hog_feat = True

TrainTestSplitSize = 0.2

import glob
import datafield as df

def datasetInit():
    images = glob.glob('dataset/*/*.png')

    for image in images:
        if 'image' in image or 'extra' in image:
            df.notcars.append(image)
        else:
            df.cars.append(image)

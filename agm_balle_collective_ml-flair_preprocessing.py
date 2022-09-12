import os
from PIL import Image

# INITIALISING START TIME AND SEED FOR RANDOM SAMPLING
print("\nStarting...")

# LOADING ML-FLAIR DATA BY GEEKSFORGEEKS
def loadData():
    print("Loading data...")
    path = 'small_images'
    os.chdir(path)
    count = 0

    from progress.bar import FillingSquaresBar
    bar = FillingSquaresBar(max=429077, suffix = '%(percent) d%% : %(elapsed)ds elapsed')

    for file in os.listdir():

        img = Image.open(file)

        if img.size == (256, 256):
            count += 1
            img = img.resize((64, 64), Image.ANTIALIAS)
            img.save("resized_" + "%s" % count + ".jpg", optimize = True, quality = 95)
        
        bar.next()
    bar.finish()

loadData()

print("Finished.\n")
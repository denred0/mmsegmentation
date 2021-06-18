import cv2
from pathlib import Path

import numpy as np

dir = Path('data/labels')
# dir = Path('mmsegmentation/iccv09Data/labels')

all_images = sorted(list(dir.glob('*.png')))
print()

for image_path in all_images:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    gray = gray.astype(int)

    np.savetxt('data/labels_txt/' + image_path.stem + '.txt', gray, fmt='%d')  # use exponential notation

#  img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
#
# # print(np.unique(img_rgb))
#  #print(img_rgb)
#
#  cv2.imwrite('data/labels_rgb/' + image_path.name, img_rgb)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 15 for Hakathon
    @author: STRH
    Crop images
"""
import cv2
import os
from natsort import natsorted

def find_contour(image):
    """
    Extract the bounding box of rgb image from the biggest contour found from total image.
    """
    mask = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _, threshInv = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Threshold Binary", threshInv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(threshInv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]  
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]  
    x,y,w,h = cv2.boundingRect(biggest_contour)

    return [x,y,w,h]

# function for cropping images
def crop(image, bbox):
    x,y,w,h = bbox
    image = image[y:y+h,x:x+w]

    return image

# apply crop to all
def apply_to_all(image_path, save_path):
    imgs_list = [f for f in os.listdir(image_path) if f.endswith('.png') or 
                        f.endswith('.PNG') or 
                        f.endswith('.webp') or 
                        f.endswith('.jpg') or 
                        f.endswith('jpeg')]
    imgs_list = natsorted(imgs_list)
    for path in imgs_list:
        img = cv2.imread(os.path.join(image_path, path))
        x,y,w,h = find_contour(img)
        # cv2.rectangle(img, (x,y),(x+w,y+h), (0,0,255),3)
        # index = imgs_list.index(path)

        img = crop(img, [x,y,w,h])
        cv2.imwrite(f"{save_path}/{path}", img)


def main():
    path = "Hackathon Official Data/images"
    save_path = "Final Data"
    apply_to_all(path, save_path)

#%% check
if __name__=="__main__":
    main()
# img = cv2.imread("Datasets/azh_wound_care_center_dataset_patches/test/images/b99276bf857375e23da3a9f5388cd6ce_0.png")
# x,y,w,h = find_contour(img)
# cv2.rectangle(img, (x,y),(x+w,y+h), (0,0,255),3)
# cv2.imshow("Threshold Binary Inverse", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
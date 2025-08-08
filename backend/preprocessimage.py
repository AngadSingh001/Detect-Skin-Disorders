import cv2

def hair_removal(img):

  

  src = img
  gray_scale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
  kernel = cv2.getStructuringElement(1, (17, 17))
  black_hat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)
  _, thresh2 = cv2.threshold(black_hat, 10, 255, cv2.THRESH_BINARY)
  dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)
  return dst

def normalize_image(img):
  normalized_img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
  return normalized_img

def image_filtering(img):
  filtered_img = cv2.GaussianBlur(img, (5, 5), 0)
  return filtered_img

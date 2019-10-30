import cv2
import numpy as np
import sys

pts_src = np.array([])
pts = 0
ptsChoose = False
image = np.array([])

def apdatePoints(action, x, y, flags, userdata):
  # Referencing global variables
  global pts_src, pts, ptsChoose, img
  # Action to be taken when left mouse button is pressed
  if action == cv2.EVENT_LBUTTONDOWN:
      ptsChoose = True
      if pts_src[0, 0] - 5 < x < pts_src[0, 0] + 5 and pts_src[0, 1] - 5 < y < pts_src[0, 1] + 5:
          pts = 0
      elif pts_src[1, 0] - 5 < x < pts_src[1, 0] + 5 and pts_src[1, 1] - 5 < y < pts_src[1, 1] + 5:
          pts = 1
      elif pts_src[2, 0] - 5 < x < pts_src[2, 0] + 5 and pts_src[2, 1] - 5 < y < pts_src[2, 1] + 5:
          pts = 2
      elif pts_src[3, 0] - 5 < x < pts_src[3, 0] + 5 and pts_src[3, 1] - 5 < y < pts_src[3, 1] + 5:
          pts = 3
      else:
          ptsChoose = False
  elif action == cv2.EVENT_LBUTTONUP:
      ptsChoose = False
  elif action == cv2.EVENT_MOUSEMOVE:
      if ptsChoose:
          pts_src = np.round(pts_src).astype(int)
          pts_src[pts] = [x, y]
          img = image.copy()
          if image.shape[0] > 700:
              img = cv2.resize(image, None, fx=700 / img.shape[0], fy=700 / img.shape[0], interpolation=cv2.INTER_CUBIC)

          ptsTmp  = pts_src.copy()
          ptsTmp[2:] = ptsTmp[2:][::-1]
          cv2.polylines(img, [ptsTmp], True, (0, 255, 0), 2, cv2.LINE_AA)
          for point in pts_src:
              cv2.circle(img, (point[0], point[1]), 10, (0, 255, 0), -1)

          cv2.imshow("Document Scanner", img)






if __name__ == "__main__" :
    # Read reference image
    refFilename = "scanned-form.jpg"
    if len(sys.argv) > 1:
        refFilename = sys.argv[1]
    img= cv2.imread(refFilename, cv2.IMREAD_COLOR)
    image = img.copy()
    # Convert images to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inr = cv2.inRange(imgGray, 200,255)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    mask = np.where((inr==255),3,2).astype('uint8')

    cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    #img = cv2.multiply(imgGray, mask)

    contours, hierarchy = cv2.findContours(cv2.multiply(imgGray, mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = np.argmax([i.shape[0]*i.shape[1] for i in (contours)])
    rect = cv2.approxPolyDP(contours[contour], 50, True)
    rect = np.squeeze(rect, axis=1)
    #cv2.drawContours(img, contours, contour, (0,255,0), 3)
    cv2.polylines(img, [rect], True, (0,255,0), 2, cv2.LINE_AA)
    rect = rect[rect[:,0].argsort()]

    if rect[-1, 1] > rect[-2, 1] and rect[1, 1] < rect[0, 1]:
        rect[:2] = rect[:2][::-1]
    elif rect[-1, 1] < rect[-2, 1] and rect[1, 1] > rect[0, 1]:
        rect[2:] = rect[2:][::-1]

    if rect[-1, 1] < rect[0, 1]:
        rect[:2] = rect[:2][::-1]
        rect[2:] = rect[2:][::-1]

    pts_dst = np.array([[0,0], [0, 645], [499, 0], [499, 645]])
    pts_src = rect

    for point in rect:
        cv2.circle(img, (point[0], point[1]), 10, (0, 255, 0), -1)
    if img.shape[0] > 700:
        img = cv2.resize(img, None, fx=700/img.shape[0], fy=700/img.shape[0], interpolation = cv2.INTER_CUBIC)
        pts_src = rect * 700 / image.shape[0]

    cv2.imshow("Document Scanner",img)
    cv2.setMouseCallback("Document Scanner", apdatePoints)

    k = cv2.waitKey(0)

    if k == 113:
        if image.shape[0] > 700:
            h, mask = cv2.findHomography(pts_src * image.shape[0] / 700, pts_dst)
        else:
            h, mask = cv2.findHomography(pts_src , pts_dst)
        imReg = cv2.warpPerspective(image, h, (500, 646))
        cv2.imshow("Warped Image",imReg)

    k = cv2.waitKey(0)

    if k == 27:  # esc to exit
        cv2.destroyAllWindows()

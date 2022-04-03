import cv2
import numpy as np
import bin.helpers

#Load image
img = cv2.imread("photo2.JPG")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Image structure copy, filled with 0's (black image with single chanell)
mask = np.zeros_like(img_gray)
img2 = cv2.imread("photo1.JPG")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialize facial landmark predictor - find features on the face
height, width, channels = img2.shape
img2_new_face = np.zeros((height, width, channels), np.uint8)

landmarks_points = bin.helpers.getLandmarksPoints(img_gray)

# Must convert to numpy array
points = np.array(landmarks_points, np.int32)
# get facial outline
convexhull = cv2.convexHull(points) 

# Place convex hull shape inside the mask 
cv2.fillConvexPoly(mask, convexhull, 255)

# Fill mask's convex hull shape with a face
face_image_1 = cv2.bitwise_and(img, img, mask=mask)

# Get rectangle around our convex hull
rect = cv2.boundingRect(convexhull) 
# To create a new empty Delaunay subdivision within our rectangle
subdiv = cv2.Subdiv2D(rect)
# Feed our detected facial landmarks to triangulate
subdiv.insert(landmarks_points)

# Get the triangle coordinates and transform to numpy array
triangles = subdiv.getTriangleList()
triangles = np.array(triangles, dtype=np.int32)

indexes_triangles = bin.helpers.performDelunay(triangles, points)
        
# Second face
landmarks_points2 = bin.helpers.getLandmarksPoints(img2_gray)

points2 = np.array(landmarks_points2, np.int32)
convexhull2 = cv2.convexHull(points2)

lines_space_mask = np.zeros_like(img_gray)
lines_space_new_face = np.zeros_like(img2)
# Triangulation of both faces
for triangle_index in indexes_triangles:
    # Triangulation of the first face
    tr1_pt1 = landmarks_points[triangle_index[0]]
    tr1_pt2 = landmarks_points[triangle_index[1]]
    tr1_pt3 = landmarks_points[triangle_index[2]]
    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
    
    rect1 = cv2.boundingRect(triangle1)
    (x, y, w, h) = rect1
    cropped_triangle = img[y: y + h, x: x + w]
    cropped_tr1_mask = np.zeros((h, w), np.uint8)


    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                       [tr1_pt2[0] - x, tr1_pt2[1] - y],
                       [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

    cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

    # Lines space
    cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, (0, 0, 255), 2)
    cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, (0, 0, 255), 2)
    cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, (0, 0, 255), 2)
    lines_space = cv2.bitwise_and(img, img, mask=lines_space_mask)
 
    # Triangulation of second face
    tr2_pt1 = landmarks_points2[triangle_index[0]]
    tr2_pt2 = landmarks_points2[triangle_index[1]]
    tr2_pt3 = landmarks_points2[triangle_index[2]]
    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

    rect2 = cv2.boundingRect(triangle2)
    (x, y, w, h) = rect2

    cropped_tr2_mask = np.zeros((h, w), np.uint8)

    points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                        [tr2_pt2[0] - x, tr2_pt2[1] - y],
                        [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

    cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

    # Warp triangles
    points = np.float32(points)
    points2 = np.float32(points2)
    M = cv2.getAffineTransform(points, points2)
    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

    # Reconstructing destination face
    img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
    img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

    img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
    img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

# Face swapped (putting 1st face into 2nd face)
img2_face_mask = np.zeros_like(img2_gray)
img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
img2_face_mask = cv2.bitwise_not(img2_head_mask)

img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
result = cv2.add(img2_head_noface, img2_new_face)

(x, y, w, h) = cv2.boundingRect(convexhull2)
center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

# Resize image
resize = bin.helpers.ResizeWithAspectRatio(seamlessclone, width=800)   
cv2.imshow("Tomas Martincic - Face swap app", resize)
cv2.waitKey(0)

cv2.destroyAllWindows()
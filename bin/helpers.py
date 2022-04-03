import cv2
import dlib

# Perform Delaunay triangulation on the face to find out in which pattern of landmarks 
# did triangulate together. Store indexes that did triangulate so we can use same facial 
# features to index both images (eg if shape_predictor_68_face_landmarks's facial features 0, 37 and 17
# triangulated we want to use that triangle for both images) - thats why we care about indexes, and not 
# about actual locations
def performDelunay(triangles, points):
    indexes_triangles = []
    for t in triangles:
        # Get three points of a given triangle
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        # Traverse all points from landmarks to find which indexes
        # have been used for given triangle (eg. 0 - facial edge + 36 - corner of eye + 31 - end of nose)
        for n in range(0, 68):
            if points[n][0] == pt1[0] and points[n][1] == pt1[1]:
                indexed1 = n
            
            if points[n][0] == pt2[0] and points[n][1] == pt2[1]:
                indexed2 = n
            
            if points[n][0] == pt3[0] and points[n][1] == pt3[1]:
                indexed3 = n

        indexes_triangles.append([indexed1, indexed2, indexed3])
    return indexes_triangles

# Function to get facial landmarks
def getLandmarksPoints(image):
    # Initialize pre-trained face detector - find location of the face
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("bin/shape_predictor_68_face_landmarks.dat")
    # Get first face from image (only first face, ignore others)
    faces = detector(image)
    first_face = faces[0]

    #get facial landmarks  
    landmarks = predictor(image, first_face)

    # Store coordinates of facial landmarks in array
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
    return landmarks_points

# Function to resize big images which keeps the aspect ratio of image
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)
import cv2
import dlib

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
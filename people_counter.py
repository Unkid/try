# USAGE
# To read and write back out to video:
# python people_counter.py -i videos/example_01.mp4 -o output/output_01.avi 
#
# To read from webcam and write back out to disk:
# python people_counter.py -o output/webcam_output.avi 
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
    help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
    help="path to optional output video file")
ap.add_argument("-f", "--fisheye", type=bool, default=False,
    help="input fisheye video")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
    help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=20,
    help="# of skip frames between detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

W = None
H = None

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
totalDown = 0
totalUp = 0
#for fisheye
if args["fisheye"]: 
    camera_matrix = np.array([[1.26125746e+03, 0.00000000e+00, 9.40592038e+02],
                             [0.00000000e+00, 1.21705719e+03, 5.96848905e+02],
                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]);
    dist_coefs = np.array([-0.49181345,  0.25848255, -0.01067125, -0.00127517, -0.01900726]);
    size = (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (size[0], size[1]), 1, (size[0], size[1]))
# start the frames per second throughput estimator
fps = FPS().start()
# loop over frames from the video stream
while True:
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame

    if args["input"] is not None and frame is None:
        break
    # if video have fisheye-lense (badly working) 
    if args["fisheye"]:
        frame = cv2.undistort(frame, camera_matrix, dist_coefs, None, newcameramtx)

    frame = imutils.resize(frame, width=650)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if W is None or H is None:
        (H, W) = frame.shape[:2]
    # if we are supposed to be writing a video to disk, initialize
    # the writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (W, H), True)

    status = "Waiting"
    rects = []
    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % args["skip_frames"] == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []
        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by requiring a minimum
            # confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])
                # if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    continue
                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)
    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"
            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()
            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))
    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)
        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)
        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            # change c[1] to c[0] for x-coordinate
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            # check to see if the object has been counted or not
            if not to.counted:
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True
                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True
        # store the trackable object in our dictionary
        trackableObjects[objectID] = to
        # draw both the ID of the object
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
        ("Status", status), ]
    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if writer is not None:
        writer.write(frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    totalFrames += 1
    fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:
    writer.release()

if not args.get("input", False):
    vs.stop()
else:
    vs.release()

cv2.destroyAllWindows()

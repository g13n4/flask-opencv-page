from flask import Flask, render_template, Response
import time
import cv2

app = Flask(__name__)
#video = cv2.VideoCapture(0)
sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor

def video_gen():
    """Video streaming generator function."""
    cap = cv2.VideoCapture('768x576.avi')

    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else: 
            break

def people_recognition_gen():
    """Video streaming generator function."""
    cap = cv2.VideoCapture('768x576.avi')

    # Read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()  # loads first frame
        if not ret: #makes it go in an endless loop
            frame = cv2.VideoCapture("768x576.avi")
            continue
        if ret:  # if there is a frame continue with code
            image = cv2.resize(frame, (0, 0), None, 1, 1)  # resizes image, doesn't really need here
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # creates gray mask
            fgmask = sub.apply(gray)  # applies the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # creates an ellipse-like kernel for image transformations
            closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel) # Removes noise in the background (part of the image proccesing)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel) # Removes noise in the objets (part of the image proccesing)
            dilation = cv2.dilate(opening, kernel) # Makes shapes bigger by "blowing" them (part of the image proccesing)
            _, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #finds contors of an image
            # defines how big/small a counter can be
            minarea = 400 
            maxarea = 50000 
            for i in range(len(contours)):  # cycles through all contours in current frame
                if hierarchy[0, i, 3] == -1:  # using hierarchy to count only parent contours (contours not within others)
                    area = cv2.contourArea(contours[i])  # defines the area of a contour
                    if minarea < area < maxarea:  # checks an area of the counter
                        cnt = contours[i] # saves current counters
                        # calculating centroids of contours
                        M = cv2.moments(cnt) 
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        # gets bounding points of contour to create rectangle
                        # x,y is top left corner and w,h is width and height
                        x, y, w, h = cv2.boundingRect(cnt)
                        # creates a rectangle around contour
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Prints centroid text in order to double check later on
                        cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,.3, (0, 0, 255), 1)
                        cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, markerSize=8, thickness=3,line_type=cv2.LINE_8)
        
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)
        #key = cv2.waitKey(20)
        #if key == 27:
        #   break
   
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(video_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/recognition_feed')
def recognition_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(people_recognition_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
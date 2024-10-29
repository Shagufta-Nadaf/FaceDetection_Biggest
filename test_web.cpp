#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace dlib;
using namespace std;

int main() {
    try {
        // Load face detection and shape prediction models
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        deserialize("/home/itstarkenn/opencv_practice/face_dlib/demo/shape_predictor_68_face_landmarks.dat") >> sp; // Ensure the path is correct

        // Initialize webcam capture
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            cerr << "ERROR: Unable to connect to camera" << endl;
            return 1;
        }

        // Create a window to display the image with landmarks
        image_window win;

        while (true) {
            cv::Mat frame;
            cap >> frame; // Capture a new frame

            if (frame.empty()) {
                cerr << "ERROR: Unable to grab from camera" << endl;
                break;
            }

            // Convert OpenCV image to Dlib format
            cv_image<bgr_pixel> dlib_frame(frame);

            // Detect faces in the image
            std::vector<rectangle> dets = detector(dlib_frame);
            cout << "Number of faces detected: " << dets.size() << endl;

            if (dets.empty()) continue;

            // Variables to track the largest face
            rectangle largest_face;
            double largest_area = 0.0;

            // Iterate through each detected face to find the largest one
            for (const auto& det : dets) {
                double area = det.width() * det.height(); // Calculate area of bounding box
                if (area > largest_area) {
                    largest_area = area;
                    largest_face = det; // Update largest face
                }
            }

            // Draw landmarks and bounding box only for the largest face
            full_object_detection shape = sp(dlib_frame, largest_face);
            for (size_t j = 0; j < shape.num_parts(); ++j) {
                int x = shape.part(j).x();
                int y = shape.part(j).y();
                draw_solid_circle(dlib_frame, point(x, y), 1.5, rgb_pixel(0, 255, 0)); // Green color for landmarks
            }
            
            win.add_overlay(largest_face, rgb_pixel(255, 0, 0)); // Red color for bounding box

            // Display the image with landmarks and bounding box
            win.set_image(dlib_frame);

            // Break the loop on 'ESC' key press
            if (cv::waitKey(30) == 27) {
                break; // Exit on ESC key
            }
        }

        cap.release(); // Release the webcam
    } catch (std::exception& e) {
        cout << "Exception: " << e.what() << endl;
    }
    return 0;
}
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace dlib;

int main() {
    try {
        // Load face detection and shape prediction models
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        deserialize("/home/itstarkenn/opencv_practice/face_dlib/demo/shape_predictor_68_face_landmarks.dat") >> sp;

        // Load input image
        array2d<rgb_pixel> img;
        load_image(img, "/home/itstarkenn/Downloads/test9.jpeg");

        // Detect faces in the image
        std::vector<rectangle> dets = detector(img);
        std::cout << "Number of faces detected: " << dets.size() << std::endl;

        if (dets.empty()) {
            std::cout << "No faces detected." << std::endl;
            return 0; // Exit if no faces are detected
        }

        // Calculate areas and find the largest bounding box
        double max_area = 0;
        rectangle largest_face;
        
        for (const auto& det : dets) {
            double area = dlib::area(det);
            if (area > max_area) {
                max_area = area;
                largest_face = det;
            }
        }

        std::cout << "Maximum area: " << max_area << std::endl;

        // Create a window to display the image with landmarks
        image_window win;

        // Detect landmarks for the largest face only
        full_object_detection shape = sp(img, largest_face);

        // Draw landmarks on the image
        for (size_t j = 0; j < shape.num_parts(); ++j) {
            int x = shape.part(j).x();
            int y = shape.part(j).y();
            draw_solid_circle(img, point(x, y), 2, rgb_pixel(0, 255, 0)); // Green color for landmarks
            std::cout << "Landmark #" << j << ": (" << x << ", " << y << ")" << std::endl;
        }

        // Display image with landmarks
        win.set_image(img);
        win.add_overlay(render_face_detections(shape));

        // Wait for a key press to exit
        std::cout << "Press Enter to exit..." << std::endl;
        std::cin.get();

    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
    return 0;
}

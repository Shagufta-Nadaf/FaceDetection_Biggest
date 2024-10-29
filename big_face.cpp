#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

using namespace dlib;
using namespace std;

int main() {
    try {
        // Load face detection and shape prediction models
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        deserialize("/home/itstarkenn/opencv_practice/face_dlib/demo/shape_predictor_68_face_landmarks.dat") >> sp;

        // Load input image
        array2d<rgb_pixel> img;
        load_image(img, "/home/itstarkenn/Downloads/demo.jpg");

        // Detect faces in the image
        std::vector<rectangle> dets = detector(img);
        cout << "Number of faces detected: " << dets.size() << endl;

        // Check if any faces were detected
        if (dets.empty()) {
            cout << "No faces found." << endl;
            return 0;
        }

        // Display detected face rectangles for debugging
        for (const auto& face : dets) {
            cout << "Detected face at: " << face << endl;
        }
// -----------------------------------------------------------------------------------
// rectangle largest_face = dets[0];
// double max_product = (largest_face.width() * largest_face.height());
//
   //     for (const auto& face : dets) {
        //    double current_product = (face.width() * face.height());
      //      cout << "Face product (width * height): " << current_product << endl;  // Print product of each detected face
       //     if (current_product > max_product) {
      //          max_product = current_product;
      //          largest_face = face;
      //      }
       // }


        // Find the largest face
        rectangle largest_face = dets[0];
        double max_area = area(largest_face);
        
        for (const auto& face : dets) {
            double current_area = area(face);
            cout << "Face area: " << current_area << endl;  // Print area of each detected face
            if (current_area > max_area) {
                max_area = current_area;
                largest_face = face;
            }
        }

        // Output the largest face's area
        cout << "Largest face area: " << max_area << endl;

        // Detect landmarks for the largest face
        full_object_detection shape = sp(img, largest_face);

        // Draw landmarks on the image
        for (size_t j = 0; j < shape.num_parts(); ++j) {
            int x = shape.part(j).x();
            int y = shape.part(j).y();
            draw_solid_circle(img, point(x, y), 2, rgb_pixel(0, 255, 0));
        }

        // Draw a rectangle around the largest face
        draw_rectangle(img, largest_face, rgb_pixel(255, 0, 0));  // Red rectangle

        // Create a window to display the image with landmarks
        image_window win;

        // Display image with landmarks for the largest face
        win.clear_overlay();
        win.set_image(img);
        win.add_overlay(render_face_detections(shape));

        cout << "Number of landmarks: " << shape.num_parts() << endl;

        // Print landmark points
        for (size_t j = 0; j < shape.num_parts(); ++j) {
            cout << "Landmark #" << j << ": " << shape.part(j) << endl;
        }

        // Wait for a key press before closing the window
        cout << "Press Enter to exit..." << endl;
        cin.get();  // Wait for user input

    } catch (std::exception& e) {
        cout << "Exception: " << e.what() << endl;
    }
    return 0;
}

#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <vector> // Include this for std::vector
#include <algorithm>

using namespace dlib; // Keep this for Dlib functions and classes
// Avoid using namespace std; 

int main() {
    try {
        // Load face detection and shape prediction models
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        deserialize("/home/itstarkenn/opencv_practice/face_dlib/demo/shape_predictor_68_face_landmarks.dat") >> sp;

        // Load input image
        array2d<rgb_pixel> img;
       // load_image(img, "/home/itstarkenn/Downloads/test5.jpeg");
        load_image(img,"/home/itstarkenn/Downloads/test1.jpeg");
        pyramid_up(img);// used because model not detecting smallest faces 
//-------------------------------------------------------------------------------------------------------------------------
        // Detect faces in the image
        std::vector<rectangle> dets = detector(img);
        std::cout << "Number of faces detected: " << dets.size() << std::endl;

        // Vector to hold areas of bounding boxes
        std::vector<double> areas(dets.size());
        
        // Calculate areas and store in vector
        for (size_t i = 0; i < dets.size(); ++i) {
            areas[i] = area(dets[i]);
            std::cout<<"dets[i]"<<dets[i]<<std::endl;
            std::cout<<"area[i]"<<areas[i]<<std::endl;
        //     double widths=dets[i].right() - dets[i].left();
        //     double heights = dets[i].bottom() - dets[i].top();
        //     std::cout << "Width : " << widths << std::endl;
        // std::cout << "Height : " << heights << std::endl;

            
        }
    
        
        // Find the index of the maximum area
        auto max_it = std::max_element(areas.begin(), areas.end());//define the range of elements,unction returns an iterator pointing to the largest element  internally use linear search
       
        int max_index = std::distance(areas.begin(), max_it);//distance=number of elements between two iterators and how far max_it from begining 
        std::cout << "max_index:"<< max_index << std::endl;
        // Get the largest bounding box
        rectangle largest_face = dets[max_index];//represents a bounding box around detected faces.
       // double max_area = *max_it;
        // double width = largest_face.right() - largest_face.left();
        // double height = largest_face.bottom() - largest_face.top();

        // std::cout << "Width of largest face: " << width << std::endl;
        // std::cout << "Height of largest face: " << height << std::endl;
        //double max_pro=width * height ;
       // std::cout <<"Maximum product:"<<max_pro <<std::endl;
        double max_area=0;
        max_area = areas.at(max_index); 
        std::cout << "Maximum area: " << max_area << std::endl;
        

        
//---------------------------------------------------------------------------------------------------------------------------
        // double width = largest_face.right() - largest_face.left();
        // double height = largest_face.bottom() - largest_face.top();

        // std::cout << "Width of largest face: " << width << std::endl;
        // std::cout << "Height of largest face: " << height << std::endl;


        // Create a window to display the image with landmarks
        image_window win;

        // Detect landmarks for the largest face only
       // full_object_detection shape = sp(img, largest_face);
       full_object_detection shape = sp(img, largest_face);
        
        // Draw landmarks on the image
        for (size_t j = 0; j < shape.num_parts(); ++j) {
            int x = shape.part(j).x();
            int y = shape.part(j).y();
            draw_solid_circle(img, point(x, y), 2, rgb_pixel(0, 255, 0)); // Green color for landmarks
            std::cout << "Landmark #" << j << ": (" << x << ", " << y << ")" << std::endl;
        }
        //------------------
        win.set_image(img);  // Set the image before adding overlays
        for (const auto& det : dets) {
            win.add_overlay(det, rgb_pixel( 0,0,255)); // Red color for bounding box
        }
        //--------------
        // Display image with landmarks
        win.set_image(img);
        win.add_overlay(render_face_detections(shape));

        // Wait for a key press to exit
        std::cout << "Press Enter to exit..." << std::endl;
        std::cin.get();

    } catch (std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
    return 0;
}
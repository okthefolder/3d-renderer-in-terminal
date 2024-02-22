#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>
#include <cmath>
#include <Windows.h>
void clear_screen() {
#ifdef _WIN32
    system("cls");
#else
    std::cout << "\033[2J\033[H";
#endif
}
struct Point3 {
    float x;
    float y;
    float z;
};

struct Point2 {
    int x;
    int y;

};

std::vector<std::vector<int>>points = { {1,1,1,0,0,0},{-1,-1,-1,-0,-0,-0} };
float x_rotation = 0;
float y_rotation = 0;
float px = 0;
float py = 0;
float pz = -10;

std::vector<std::vector<int>> make_cuboid(int x1, int y1, int z1, int x2, int y2, int z2) {
    std::vector<std::vector<int>> vertices;
    // Define the eight vertices of the cuboid
    vertices.push_back({ x1, y1, z1 }); // 0
    vertices.push_back({ x2, y1, z1 }); // 1
    vertices.push_back({ x2, y2, z1 }); // 2
    vertices.push_back({ x1, y2, z1 }); // 3
    vertices.push_back({ x1, y1, z2 }); // 4
    vertices.push_back({ x2, y1, z2 }); // 5
    vertices.push_back({ x2, y2, z2 }); // 6
    vertices.push_back({ x1, y2, z2 }); // 7

    return vertices;

}

void cuboid_to_vertices(std::vector<std::vector<int>>& points){
    std::vector<std::vector<int>> vertices;
    for (std::vector<int> p : points) {
        for (std::vector<int> vertice : make_cuboid(p[0], p[1], p[2], p[3], p[4], p[5])) {
            vertices.push_back(vertice);
        }
    }
    points = vertices;
}

void add_rotation(float x_rotation, float y_rotation, float& x_co, float& y_co, float& z_co){
    float new_x_co = x_co*std::cos(x_rotation)-z_co*std::sin(x_rotation);
    float new_z_co = x_co * std::sin(x_rotation) + z_co * std::cos(x_rotation);
    float new_y_co = y_co * std::sin(y_rotation) + new_z_co * std::cos(y_rotation);
    new_z_co = y_co * std::cos(y_rotation) - new_z_co * std::sin(y_rotation);
    if (new_z_co != 0) {
        x_co = (new_x_co / new_z_co)*100;
        y_co = (new_y_co / new_z_co)*100;
    }
    else {
        x_co = 999999;
        y_co = 999999;
    }
}

void draw_screen(const std::vector<int>& screen) {
    std::string buffer; // Buffer to hold the entire screen content
    buffer.reserve(156 * 39 * 3); // Reserve space for the entire content

    for (int i = 0; i < 156 * 39; ++i) {
        if (screen[i] != 0) {
            buffer += std::to_string(screen[i]); // Append each number to the buffer
        }
        else {
            buffer += "_"; // Append each number to the buffer
        }
    }
    clear_screen();
    std::cout << buffer; // Print the entire buffer at once
    std::cout.flush(); // Ensure output is immediately visible
}

bool isPointInFrontOfCamera(float angleX, float angleY, Point3 point) {
    // Calculate camera's viewing direction vector
    float vx = cos(angleY) * sin(angleX);
    float vy = sin(angleY);
    float vz = cos(angleY) * cos(angleX);

    // Compute dot product
    float dotProduct = point.x * vx + point.y * vy + point.z * vz;

    // If dot product is positive, point is in front of camera
    return dotProduct >= 0;
}

void order_points(Point2& a, Point2& b, Point2& c){
    if (a.y < b.y) std::swap(a, b);
    if (a.y < c.y) std::swap(a, c);
    if (b.y < c.y) std::swap(b, c);
}

std::vector<std::vector<int>> rasterize(Point2 a, Point2 b, Point2 c) {
    std::vector<std::vector<int>> rasterized;
    std::vector<int> x_co_for_lines_1;
    std::vector<int> x_co_for_lines_2;
    order_points(a, b, c);
    std::swap(c, a);
    //std::swap(b, a);
    //std::cout << c.y << " " << b.y << " " << a.y << std::endl;
    //c>b>a
    for (int i = c.y; i > b.y; i--) {
        
        if (b.y - c.y != 0) {
            x_co_for_lines_2.push_back((i - c.y) * (b.x - c.x) / (b.y - c.y) + c.x);
        }
        else {
            x_co_for_lines_2.push_back(b.x);
        }
    }
    for (int i = b.y; i >= a.y; i--) {
        if (b.y - a.y != 0) {
            x_co_for_lines_2.push_back((i - b.y) * (a.x - b.x) / (a.y - b.y) + b.x);
        }
        else {
            x_co_for_lines_2.push_back(a.x);
        }
    }
    for (int i = c.y; i >= a.y; i--) {
        if (c.y - a.y != 0) {
            x_co_for_lines_1.push_back((i - c.y) * (a.x - c.x) / (a.y - c.y) + c.x);
        }
        else {
            x_co_for_lines_1.push_back(c.x);
        }
    }
    
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    for (int i = 0; i < x_co_for_lines_1.size();i++) {
        if (x_co_for_lines_1[i] < x_co_for_lines_2[i]) {
            for (int x = x_co_for_lines_1[i]; x <= x_co_for_lines_2[i]; x++) {
                rasterized.push_back({ x, c.y-i });
                
            }
        }
        else {
            for (int x = x_co_for_lines_2[i]; x <= x_co_for_lines_1[i]; x++) {
                rasterized.push_back({ x, c.y-i });
            }
        }
    }
    //std::cout << rasterized.size() << std::endl;
    return rasterized;
}

void update_screen(std::vector<int>& screen, std::vector<std::vector<int>> points, float x_rotation, float y_rotation, float px, float py, float pz) {
    // Simulate some changes to the screen content
    std::vector<Point2> new_points;
    for (int i = 0; i < 40 * 156;i++) {
        screen[i] = 0;
    }
    for (std::vector<int> point : points) {
        float p_x_co = point[0]-px;
        float p_y_co = point[1]-py;
        float p_z_co = point[2]-pz;
        Point3 point = { p_x_co, p_y_co, p_z_co };
        if (isPointInFrontOfCamera(x_rotation, y_rotation, point)){
            //std::cout << floor(sqrt(p_x_co*p_x_co+p_z_co*p_z_co)/(p_z_co)) << " " << floor(1/std::cos(x_rotation)) << std::endl;
            add_rotation(x_rotation, y_rotation - 3.14 / 2, p_x_co, p_y_co, p_z_co);
            //std::cout << (fmod(x_rotation, 3.14 / 2)) << " " << std::atan((point[2] - pz) / (point[0] - px)) << " " << fmod( x_rotation, 3.14 / 2)-3.14/2 << std::endl;

            p_x_co += 78;
            p_y_co += 20;
            Point2 point2 = { static_cast<int>(p_x_co),static_cast<int>(p_y_co) };
            new_points.push_back(point2);
            //std::cout << p_x_co << " " << p_y_co << std::endl;
            if (p_y_co < 40 && p_y_co > 0 && p_x_co > 0 && p_x_co < 156) {
                screen[156 * floor(p_y_co) + p_x_co] = 1;
            }
        }
    }

    std::vector<std::vector<int>> faces = {
        // Front face
        {0, 1, 2},
        {2, 3, 0},
        // Back face
        {5, 4, 7},
        {7, 6, 5},
        // Top face
        {3, 2, 6},
        {6, 7, 3},
        // Bottom face
        {4, 5, 1},
        {1, 0, 4},
        // Left face
        {4, 0, 3},
        {3, 7, 4},
        // Right face
        {1, 5, 6},
        {6, 2, 1}
    };
    for (int i = 0; i < points.size() / 8; i++) {
        for (std::vector<int> triangle : faces) {
            std::vector<std::vector<int>> rasterized_points = rasterize(new_points[triangle[0] + 8 * i], new_points[triangle[1] + 8 * i], new_points[triangle[2] + 8 * i]);
            for (std::vector<int> p : rasterized_points) {
                int p_x_co = p[0];
                int p_y_co = p[1];
                //std::cout << p_x_co << " " << p_y_co << std::endl;
                if (p_y_co < 40 && p_y_co > 0 && p_x_co > 0 && p_x_co < 156) {
                    //std::cout << "f" << std::endl;
                    screen[156 * floor(p_y_co) + p_x_co] = 1;
                }
            }
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void controls(float& x_rotation, float& y_rotaion, float& px, float& py, float& pz) {
    if (GetAsyncKeyState(VK_UP) & 0x8000) {
        y_rotation += 0.1;
    }
    if (GetAsyncKeyState(VK_DOWN) & 0x8000) {
         y_rotation -= 0.1;
    }
    if (GetAsyncKeyState(VK_LEFT) & 0x8000) {
        x_rotation += 0.1;
    }
    if (GetAsyncKeyState(VK_RIGHT) & 0x8000) {
        x_rotation -= 0.1;
    }

    if (GetAsyncKeyState(VK_SPACE) & 0x8000) {
        py += 1;
    }
    if (GetAsyncKeyState('C') & 0x8000) {
        py -= 1;
    }

    if (GetAsyncKeyState('W') & 0x8000) { // Move forward
        px += -1*std::sin(-x_rotation);
        pz -= -1 * std::cos(x_rotation);
    }
    if (GetAsyncKeyState('S') & 0x8000) { // Move backward
        px -= -1 * std::sin(-x_rotation);
        pz += -1 * std::cos(x_rotation);
    }
    if (GetAsyncKeyState('A') & 0x8000) { // Turn left
        px += -1 * std::cos(-x_rotation);
        pz -= -1 * std::sin(x_rotation);
    }
    if (GetAsyncKeyState('D') & 0x8000) { // Turn right
        px -= -1 * std::cos(-x_rotation);
        pz += -1 * std::sin(x_rotation);
    }
    //Sleep(100);
}

int main() {
    std::vector<int> screen(156 * 40);
    cuboid_to_vertices(points);
    // Seed random number generator
    srand(static_cast<unsigned int>(time(nullptr)));

    while (true) {
        controls(x_rotation, y_rotation,px,py,pz);
        update_screen(screen, points, x_rotation, y_rotation, px, py, pz); // Update screen content
        draw_screen(screen);   // Draw updated screen
        //std::cout << px << " " << py << " " << pz << std::endl;
        //std::cout << y_rotation << std::endl;
        //std::cout << x_rotation << std::endl;

        // Sleep for a short duration to control the frame rate
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}

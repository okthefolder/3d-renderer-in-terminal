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
    int z;

};

std::vector<std::vector<int>>points = { {1,1,1,0,0,0}};
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
    /*if (new_z_co != 0) {
        x_co = (new_x_co / new_z_co)*100;
        y_co = (new_y_co / new_z_co)*100;
    }
    else {
        x_co = 999999;
        y_co = 999999;
    }*/
    x_co = new_x_co;
    y_co = new_y_co;
    z_co = new_z_co;
}

void draw_screen(const std::vector<std::vector<int>>& screen) {
    std::string buffer; // Buffer to hold the entire screen content
    buffer.reserve(156 * 39 * 3); // Reserve space for the entire content

    for (int i = 0; i < 156 * 39; ++i) {
        if (screen[i][0] != 0) {
            buffer += std::to_string(screen[i][0]); // Append each number to the buffer
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
    std::vector<int> z_co_for_lines_1;
    std::vector<int> z_co_for_lines_2;
    order_points(a, b, c);
    std::swap(c, a);
    for (int i = c.y; i > b.y; i--) {

        if (b.y - c.y != 0) {
            x_co_for_lines_2.push_back((i - c.y) * (b.x - c.x) / (b.y - c.y) + c.x);
            z_co_for_lines_2.push_back((i - c.y) * (b.z - c.z) / (b.y - c.y) + c.z);
        }
        else {
            x_co_for_lines_2.push_back(b.x);
            z_co_for_lines_2.push_back(b.z);
        }
    }
    for (int i = b.y; i >= a.y; i--) {
        if (b.y - a.y != 0) {
            x_co_for_lines_2.push_back((i - b.y) * (a.x - b.x) / (a.y - b.y) + b.x);
            z_co_for_lines_2.push_back((i - b.y) * (a.z - b.z) / (a.y - b.y) + b.z);
        }
        else {
            x_co_for_lines_2.push_back(a.x);
            z_co_for_lines_2.push_back(a.z);
        }
    }
    for (int i = c.y; i >= a.y; i--) {
        if (c.y - a.y != 0) {
            x_co_for_lines_1.push_back((i - c.y) * (a.x - c.x) / (a.y - c.y) + c.x);
            z_co_for_lines_1.push_back((i - c.y) * (a.z - c.z) / (a.y - c.y) + c.z);
        }
        else {
            x_co_for_lines_1.push_back(c.x);
            z_co_for_lines_1.push_back(c.z);
        }
    }

    for (int i = 0; i < x_co_for_lines_1.size(); i++) {
        if (x_co_for_lines_1[i] < x_co_for_lines_2[i]) {
            for (int x = x_co_for_lines_1[i]; x <= x_co_for_lines_2[i]; x++) {
                //ADD Z COORDINATE
                if (x_co_for_lines_1[i] != x_co_for_lines_2[i]) {
                    rasterized.push_back({ x, c.y - i, z_co_for_lines_1[i] + (x - x_co_for_lines_1[i]) * (z_co_for_lines_1[i] - z_co_for_lines_2[i]) / (x_co_for_lines_1[i] - x_co_for_lines_2[i]) });
                }
                else {
                    rasterized.push_back({ x, c.y - i, z_co_for_lines_1[i] });
                }
            }
        }
        else {
            for (int x = x_co_for_lines_2[i]; x <= x_co_for_lines_1[i]; x++) {
                if (x_co_for_lines_1[i] != x_co_for_lines_2[i]) {
                    rasterized.push_back({ x, c.y - i, z_co_for_lines_2[i] + (x - x_co_for_lines_2[i]) * (z_co_for_lines_1[i] - z_co_for_lines_2[i]) / (x_co_for_lines_1[i] - x_co_for_lines_2[i]) });
                }
                else {
                    rasterized.push_back({ x, c.y - i, z_co_for_lines_1[i] });
                }
            }
        }
    }
    return rasterized;
}

void update_screen(std::vector<std::vector<int>>& screen, std::vector<std::vector<int>> points, float x_rotation, float y_rotation, float px, float py, float pz) {
    // Simulate some changes to the screen content
    std::vector<Point2> new_points;
    for (int i=0; i < points.size(); i++) {
        new_points.push_back({0,0,0});
    }
    for (int i = 0; i < 40 * 156;i++) {
        screen[i] = { 0,1000000000 };
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
    int how_many_triangles=0;
    
    for (int i = 0; i < points.size() / 8; i++) {
        for (std::vector<int> triangle : faces) {
            float p_x_co1 = points[i*8 + triangle[0]][0] - px;
            float p_y_co1 = points[i * 8 + triangle[0]][1] - py;
            float p_z_co1 = points[i * 8 + triangle[0]][2] - pz;
            
            float p_x_co2 = points[i * 8 + triangle[1]][0] - px;
            float p_y_co2 = points[i * 8 + triangle[1]][1] - py;
            float p_z_co2 = points[i * 8 + triangle[1]][2] - pz;

            float p_x_co3 = points[i * 8 + triangle[2]][0] - px;
            float p_y_co3 = points[i * 8 + triangle[2]][1] - py;
            float p_z_co3 = points[i * 8 + triangle[2]][2] - pz;

            Point3 point1 = { p_x_co1, p_y_co1, p_z_co1 };
            Point3 point2 = { p_x_co2, p_y_co2, p_z_co2 };
            Point3 point3 = { p_x_co3, p_y_co3, p_z_co3 };
            if (isPointInFrontOfCamera(x_rotation, y_rotation, point1)&& isPointInFrontOfCamera(x_rotation, y_rotation, point2)&&isPointInFrontOfCamera(x_rotation, y_rotation, point3)) {
                how_many_triangles++;
                add_rotation(x_rotation, y_rotation - 3.14 / 2, p_x_co1, p_y_co1, p_z_co1);
                add_rotation(x_rotation, y_rotation - 3.14 / 2, p_x_co2, p_y_co2, p_z_co2);
                add_rotation(x_rotation, y_rotation - 3.14 / 2, p_x_co3, p_y_co3, p_z_co3);
                if (   static_cast<int>(p_z_co1)* static_cast<int>(p_z_co2)* static_cast<int>(p_z_co3) == 0) {
                    std::cout << "weuignriugnreiae" << std::endl;
                }
                new_points[i*8+triangle[0]]={ static_cast<int>(100 * p_x_co1),static_cast<int>(100 * p_y_co1), static_cast<int>(100 * p_z_co1) };
                new_points[i * 8 + triangle[1]] = { static_cast<int>(100 * p_x_co2),static_cast<int>(100 * p_y_co2), static_cast<int>(100 * p_z_co2) };
                new_points[i * 8 + triangle[2]] = { static_cast<int>(100 * p_x_co3),static_cast<int>(100 * p_y_co3), static_cast<int>(100 * p_z_co3) };
            }
            else {
                std::cout << "e" << std::endl;
            }
        }
    }
    std::cout << new_points.size() << std::endl;
    if (new_points.size() != 0) {
        for (int i = 0; i < new_points.size() / 8; i++) {
            int loop_index = 0;
            for (std::vector<int> triangle : faces) {
                    std::vector<std::vector<int>> rasterized_points = rasterize(new_points[8*i+triangle[0]], new_points[8*i+triangle[1]], new_points[8*i+triangle[2]]);
                    std::cout << rasterized_points.size();
                    int triangle_index = 0;
                    if (rasterized_points.size()) {
                        loop_index++;
                    }
                    for (std::vector<int> p : rasterized_points) {
                        
                        int p_x_co = p[0];
                        int p_y_co = p[1];
                        int p_z_co = p[2];
                        if (p_z_co != 0 && abs(100 * p_y_co / p_z_co)<20 && abs(100 * p_x_co / p_z_co)<78) {
                            if (p_z_co < screen[156 * floor(100 * p_y_co / p_z_co) + 20 * 156 + 78 + 100 * p_x_co / p_z_co][1]) {
                                screen[156 * floor(100 * p_y_co / p_z_co) + 20 * 156 + 78 + 100 * p_x_co / p_z_co] = { triangle[0]+1,p_z_co};
                            }
                        }
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
    std::vector<std::vector<int>> screen(156 * 40);
    cuboid_to_vertices(points);
    // Seed random number generator
    srand(static_cast<unsigned int>(time(nullptr)));

    while (true) {
        controls(x_rotation, y_rotation,px,py,pz);
        update_screen(screen, points, x_rotation, y_rotation, px, py, pz); // Update screen content
        draw_screen(screen);   // Draw updated screen
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>
#include <cmath>
#include <Windows.h>
#include <algorithm>
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
    float x;
    float y;
    float z;

};

const int characters_per_row = 1880;
const int number_of_columns = 480;

std::vector<char> characters = {
        'W', '@', 'M', '#', 'X', 'A', 'B', '8', '&', '%', '+', '=', 'i', 'l', '1'
};

std::vector<std::vector<int>>points = { 
    {0, 0, 0, 2, 2, 2},
    {-1, -1, -1, 0, 0, 0},
    {1, 1, 1, 3, 3, 3},
    {3, 3, 3, 5, 5, 5},
    {-2, -2, -2, -1, -1, -1},
    {2, 2, 2, 4, 4, 4},
    {-3, -3, -3, -2, -2, -2},
    {0, 0, 0, 1, 1, 1},
    {5, 5, 5, 6, 6, 6},
    {-1, -1, -1, 1, 1, 1},
    {0, 0, 0, 3, 3, 3},
    {-2, -2, -2, 2, 2, 2},
    {1, 1, 1, 4, 4, 4},
    {-3, -3, -3, 1, 1, 1},
    {2, 2, 2, 5, 5, 5} };
float x_rotation = 0;
float y_rotation = 0;
float px = 0;
float py = 0;
float pz = -10.1;

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
    buffer.reserve(characters_per_row * number_of_columns * 3); // Reserve space for the entire content
    //std::cout << "buf" << std::endl;
    //std::cout << screen.size() << std::endl;
    //std::cout << screen[6241][0] << std::endl;
    for (int i = 0; i < (characters_per_row) * number_of_columns; ++i) {
        //std::cout << i << std::endl;
        if (screen[i][0] != 0) {
          //  std::cout << "k" << std::endl;
            buffer += characters[screen[i][0]]; // Append each number to the buffer
            //std::cout << "j" << std::endl;
        }
        else {
            //std::cout << "o" << std::endl;
            buffer += "_"; // Append each number to the buffer
           // std::cout << "i" << std::endl;
        }
    }
    //std::cout << "buf" << std::endl;
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

std::tuple<Point2, Point2> calculateTriangleBoundingBox(const Point2& a, const Point2& b, const Point2& c) {
    // Find minimum and maximum x coordinates
    float minX = min( a.x, b.x, c.x );
    float maxX = max( a.x, b.x, c.x );

    // Find minimum and maximum y coordinates
    float minY = min( a.y, b.y, c.y );
    float maxY = max( a.y, b.y, c.y );

    // Create bounding box points
    Point2 topLeft = { minX, minY };
    Point2 bottomRight = { maxX, maxY };
    return std::make_tuple( topLeft, bottomRight );
}

void order_points(Point2& a, Point2& b, Point2& c){
    if (a.y < b.y) std::swap(a, b);
    if (a.y < c.y) std::swap(a, c);
    if (b.y < c.y) std::swap(b, c);
}

bool is_point_inside_screen(Point2 point) {
    if (abs(point.x)>=characters_per_row/2 || abs(point.y) >= number_of_columns/2) {
        return false;
    }
    return true;
}

std::vector<Point2> intersection(Point2 p1, Point2 p2) {
    std::vector<Point2> intersections;
    
    //FIX Z VALUE
    Point2 l1_intersection = { -999999, -999999,0 };
    Point2 l2_intersection = { -999999, -999999,0 };
    Point2 l3_intersection = { -999999, -999999,0 };
    Point2 l4_intersection = { -999999, -999999,0 };
    if (p1.y != p2.y) {
        float t1 = (-(number_of_columns - 1) / 2 - p1.y) / (p2.y - p1.y);
        float z_1 = p1.z+t1*(p2.z-p1.z);

        float t2 = ((number_of_columns - 1) / 2 - p1.y) / (p2.y - p1.y);
        float z_2 = p1.z + t2 * (p2.z - p1.z);

        l1_intersection = { ((-(number_of_columns - 1) / 2 - p1.y) * ((p1.x - p2.x) / (p1.y - p2.y)) + p1.x), -(number_of_columns-1) / 2, z_1 };
        l2_intersection = { (((number_of_columns - 1) / 2 - p1.y) * ((p1.x - p2.x) / (p1.y - p2.y)) + p1.x), (number_of_columns - 1) / 2, z_2 };
        //std::cout << "l1 " << l1_intersection.x << " " << l1_intersection.y << std::endl;
    }
    if (p1.x != p2.x){

        float t3 = (-(characters_per_row - 1) / 2 - p1.x) / (p2.x - p1.x);
        float z_3 = p1.z + t3 * (p2.z - p1.z);

        float t4 = ((characters_per_row - 1) / 2 - p1.x) / (p2.x - p1.x);
        float z_4 = p1.z + t4 * (p2.z - p1.z);

        l3_intersection = { (-(characters_per_row - 1) / 2),   (-(characters_per_row - 1) / 2 - p1.x) * ((p1.y - p2.y) / (p1.x - p2.x)) + p1.y,  z_3 };
        //std::cout << "l3.x: " << l3_intersection.x<<" chp: "<< << std::endl;
        l4_intersection = { (characters_per_row - 1) / 2,   ((characters_per_row - 1) / 2 - p1.x) * ((p1.y - p2.y) / (p1.x - p2.x)) + p1.y, z_4 };
        //std::cout << -(characters_per_row-1)/2<<" " << "l3:" << l3_intersection.x << " " << l3_intersection.y << std::endl;
       
    }
    if (l1_intersection.x<=max(p1.x, p2.x) && l1_intersection.x >= min(p1.x, p2.x) && l1_intersection.y <= max(p1.y, p2.y) && l1_intersection.y >= min(p1.y, p2.y) && abs(l1_intersection.x)<characters_per_row/2 && abs(l1_intersection.y)<number_of_columns / 2 && l1_intersection.z!=0) {
        intersections.push_back(l1_intersection); 
    }
    if (l2_intersection.x<=max(p1.x, p2.x) && l2_intersection.x >= min(p1.x, p2.x) && l2_intersection.y <= max(p1.y, p2.y) && l2_intersection.y >= min(p1.y, p2.y) && abs(l2_intersection.x) < characters_per_row / 2 && abs(l2_intersection.y) < number_of_columns / 2 && l2_intersection.z != 0) {
        intersections.push_back(l2_intersection);
    }
    if (l3_intersection.y<=max(p1.y, p2.y) && l3_intersection.y >= min(p1.y, p2.y) && l3_intersection.x <= max(p1.x, p2.x) && l3_intersection.x >= min(p1.x, p2.x) && abs(l3_intersection.x) < characters_per_row / 2 && abs(l3_intersection.y) < number_of_columns / 2 && l3_intersection.z != 0) {
        intersections.push_back(l3_intersection);
    }



    if (l4_intersection.y<=max(p1.y, p2.y)+1 && l4_intersection.y >= min(p1.y, p2.y)-1 && l4_intersection.x <= max(p1.x, p2.x) && l4_intersection.x >= min(p1.x, p2.x) && abs(l4_intersection.x) <= characters_per_row / 2 && abs(l4_intersection.y) < number_of_columns / 2 && l4_intersection.z != 0) {
        intersections.push_back(l4_intersection);
    }

    //std::cout << "intersect p1.x:" << p1.x << " p1.y:" << p1.y << " p2.x:" << p2.x << " p2.y:" << p2.y << " intersec:" <<l1_intersection.x<< std::endl;
    return intersections;
}

bool intpoint_inside_trigon(Point2 s, Point2 a, Point2 b, Point2 c)
{
    int as_x = s.x - a.x;
    int as_y = s.y - a.y;

    bool s_ab = (b.x - a.x) * as_y - (b.y - a.y) * as_x > 0;

    if ((c.x - a.x) * as_y - (c.y - a.y) * as_x > 0 == s_ab) {
        return false;
    }
    if ((c.x - b.x) * (s.y - b.y) - (c.y - b.y) * (s.x - b.x) > 0 != s_ab) {
        return false;
    }
    return true;
}

bool compareX(const Point2& p1, const Point2& p2) {
    return p1.x < p2.x;
}

bool compareAngles(const Point2& p1, const Point2& p2, const Point2& centroid) {
    // Calculate angles using atan2
    float angle1 = atan2(p1.y - centroid.y, p1.x - centroid.x);
    float angle2 = atan2(p2.y - centroid.y, p2.x - centroid.x);
    return angle1 < angle2;
}

std::vector<float> plane_equation(Point2 p1, Point2 p2, Point2 p3) {
    // Calculate vectors v1 and v2
    float v1x = p2.x - p1.x;
    float v1y = p2.y - p1.y;
    float v1z = p2.z - p1.z;

    float v2x = p3.x - p1.x;
    float v2y = p3.y - p1.y;
    float v2z = p3.z - p1.z;

    // Calculate the cross product to find the normal vector
    float nx = v1y * v2z - v1z * v2y;
    float ny = v1z * v2x - v1x * v2z;
    float nz = v1x * v2y - v1y * v2x;

    // Calculate D using the first point
    float D = nx * p1.x + ny * p1.y + nz * p1.z;

    return { nx, ny, nz, -D };
}

std::vector<std::vector<float>> rasterize(Point2 a, Point2 b, Point2 c,int test) {
    std::vector<std::vector<float>> rasterized;
    std::vector<float> x_co_for_lines_1;
    std::vector<float> x_co_for_lines_2;
    std::vector<float> z_co_for_lines_1;
    std::vector<float> z_co_for_lines_2;
    order_points(a, b, c);
    std::swap(c, a);
    if (max(abs(a.x), abs(b.x), abs(c.x)) >     characters_per_row / 2 || (max(abs(a.y), abs(b.y), abs(c.y)) > number_of_columns / 2)) {
        //std::cout << "gg" << std::endl;
        //std::cout << a.x << " a& " << a.y << std::endl;
        //std::cout << b.x << " b& " << b.y << std::endl;
        //std::cout << c.x << " c& " << c.y << std::endl;
            std::vector<Point2> a_b_intersection=intersection(a, b);
            std::vector<Point2> a_c_intersection = intersection(a, c);
            std::vector<Point2> b_c_intersection = intersection(b, c);
            std::vector<Point2> points_for_triangulation;
            if (is_point_inside_screen(a)) {
                points_for_triangulation.push_back(a);
         //       std::cout << a.x << " a " << a.y << std::endl;
            };
            if (is_point_inside_screen(b)) { 
                points_for_triangulation.push_back(b);
           //     std::cout << b.x << " b " << b.y << std::endl;
            };
            if (is_point_inside_screen(c)) { 
                points_for_triangulation.push_back(c);
               // std::cout << c.x << " c " << c.y << std::endl;
            };
            for (Point2 p : a_b_intersection) {
               // std::cout << p.x << " a_b " << p.y <<" " <<p.z<< std::endl;
                points_for_triangulation.push_back(p);
            }
            for (Point2 p : a_c_intersection) {
                //std::cout << p.x << " a_c " << p.y << " " << p.z << std::endl;
                points_for_triangulation.push_back(p);
            }
            for (Point2 p : b_c_intersection) {
                //std::cout << p.x << " b_c " << p.y << " " << p.z << std::endl;
                points_for_triangulation.push_back(p);
            }
            std::vector<Point2> screen_vertices = { {characters_per_row / 2 -1,number_of_columns / 2-1},{-characters_per_row / 2+1,-number_of_columns / 2+1},{characters_per_row / 2-1,-number_of_columns / 2+1},{-characters_per_row / 2+1,number_of_columns / 2 + 1} };
            for (int i = 0; i < 4; ++i) {
                if (intpoint_inside_trigon(screen_vertices[i], a, b, c)) {
                    std::vector<float> plane_coefficients=plane_equation(a, b, c);
                    float p_z = -(plane_coefficients[3]+ plane_coefficients[0]* screen_vertices[i].x + plane_coefficients[1]* screen_vertices[i].y) / plane_coefficients[2];
                   // std::cout << "SC" << screen_vertices[i].x << " " << screen_vertices[i].y << " " << p_z << std::endl;
                    points_for_triangulation.push_back({screen_vertices[i].x,screen_vertices[i].y, p_z});
                }
            }
            if (points_for_triangulation.size() != 0) {
                float centroidx = 0;
                float centroidy = 0;
                //std::cout << points_for_triangulation.size() << std::endl;
                for (Point2 p : points_for_triangulation) {
                    
                    centroidx += p.x;
                    centroidy += p.y;
                }
                for (int i = 0; i < points_for_triangulation.size(); i++) {
                    //std::cout << "(" << points_for_triangulation[i].x << ", " << points_for_triangulation[i].y << ")" << std::endl;
                }
                Point2 centroid = { centroidx / (points_for_triangulation.size()),centroidy / (points_for_triangulation.size()) ,1 };
                std::sort(points_for_triangulation.begin(), points_for_triangulation.end(), [&](const Point2& p1, const Point2& p2) {
                    return compareAngles(p1, p2, centroid);
                    });
                //std::cout <<"polygon size:" << points_for_triangulation.size() << std::endl;
                //std::cout << "centre:" << centroid.x << " " << centroid.y << std::endl;
                for (int i = 0; i < points_for_triangulation.size(); i++) {
                    //std::cout << "(" << points_for_triangulation[i].x << ", " << points_for_triangulation[i].y << ")" << std::endl;
                }
                for (int i = 1; i < points_for_triangulation.size() - 1; i++) {
                    
                    //std::cout << points_for_triangulation[0].y << " " << points_for_triangulation[i].y << " " << points_for_triangulation[i + 1].y<< std::endl;
                    for (std::vector<float> tri_rasteri : rasterize(points_for_triangulation[0], points_for_triangulation[i], points_for_triangulation[i + 1],1)) {
                        //std::cout << "giroi" << std::endl;
                        rasterized.push_back(tri_rasteri);
                    }
                }
            }
            else {
                //for (int i = 0; i < 1000; i++) {
                //    std::cout << "fjesigojseo" << std::endl;
                //}
            }
            //std::cout << rasterized.size() << std::endl;;
            return rasterized;
            //std::sort(points_for_triangulation.begin(), points_for_triangulation.end(), compareX);
            



    }
    //std::cout << a.x << " " << a.y<<" " << b.x << " " << b.y << " " << c.x << " " << c.y << std::endl;
    for (float i = c.y; i > b.y; i--) {

        if (b.y - c.y != 0) {
            x_co_for_lines_2.push_back((i - c.y) * (b.x - c.x) / (b.y - c.y) + c.x);
            z_co_for_lines_2.push_back((i - c.y) * (b.z - c.z) / (b.y - c.y) + c.z);
        }
        else {
            x_co_for_lines_2.push_back(b.x);
            z_co_for_lines_2.push_back(b.z);
        }
    }
    for (float i = b.y; i >= a.y; i--) {
        if (b.y - a.y != 0) {
            x_co_for_lines_2.push_back((i - b.y) * (a.x - b.x) / (a.y - b.y) + b.x);
            z_co_for_lines_2.push_back((i - b.y) * (a.z - b.z) / (a.y - b.y) + b.z);
        }
        else {
            x_co_for_lines_2.push_back(a.x);
            z_co_for_lines_2.push_back(a.z);
        }
    }
    for (float i = c.y; i >= a.y; i--) {
        if (c.y - a.y != 0) {
            x_co_for_lines_1.push_back((i - c.y) * (a.x - c.x) / (a.y - c.y) + c.x);
            z_co_for_lines_1.push_back((i - c.y) * (a.z - c.z) / (a.y - c.y) + c.z);
        }
        else {
            x_co_for_lines_1.push_back(c.x);
            z_co_for_lines_1.push_back(c.z);
        }
    }

    for (float i = 0; i < x_co_for_lines_1.size(); i++) {
        if (x_co_for_lines_1[i] < x_co_for_lines_2[i]) {
            for (float x = x_co_for_lines_1[i]; x <= x_co_for_lines_2[i]; x++) {
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
            for (float x = x_co_for_lines_2[i]; x <= x_co_for_lines_1[i]; x++) {
                if (x_co_for_lines_1[i] != x_co_for_lines_2[i]) {
                    rasterized.push_back({ x, c.y - i, z_co_for_lines_2[i] + (x - x_co_for_lines_2[i]) * (z_co_for_lines_1[i] - z_co_for_lines_2[i]) / (x_co_for_lines_1[i] - x_co_for_lines_2[i]) });
                }
                else {
                    rasterized.push_back({ x, c.y - i, z_co_for_lines_1[i] });
                }
            }
        }
    }
    //std::cout << rasterized.size() << std::endl;
    if (test == 1) {
       // rasterized.push_back({ 0,0 });
    }
    return rasterized;
}

void update_screen(std::vector<std::vector<int>>& screen, std::vector<std::vector<int>> points, float x_rotation, float y_rotation, float px, float py, float pz) {
    // Simulate some changes to the screen content
    std::vector<Point2> new_points;
    for (int i=0; i < points.size(); i++) {
        new_points.push_back({0,0,0});
    }
    for (int i = 0; i < characters_per_row*number_of_columns;i++) {
        screen[i] = { 0,1000000000 };
    }




    std::vector<std::vector<int>> faces = {
        // Front face
        {0, 1, 2,1},
        {2, 3, 0,1},
        // Back face
        
        {5, 4, 7,4},
        {7, 6, 5,4},
        // Top face
        {3, 2, 6,7},
        {6, 7, 3,7},
        // Bottom face
        {4, 5, 1,9},
        {1, 0, 4,9},
        // Left face
        {4, 0, 3,12},
        {3, 7, 4,12},
        // Right face
        {1, 5, 6,14},
        {6, 2, 1,14}
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
                if (static_cast<int>(100 * p_z_co1) != 0) {
                    new_points[i * 8 + triangle[0]] = { (number_of_columns * p_x_co1 / p_z_co1),(number_of_columns * p_y_co1 / p_z_co1), (number_of_columns * p_z_co1) };
                }
                else {
                    //new_points[i * 8 + triangle[0]] = { static_cast<int>(100 * p_x_co1),static_cast<int>(100 * p_y_co1), static_cast<int>(100 * p_z_co1) };
                }
                if (static_cast<int>(100 * p_z_co2) != 0) {
                    new_points[i * 8 + triangle[1]] = { (number_of_columns * p_x_co2 / p_z_co2),(number_of_columns * p_y_co2 / p_z_co2), (number_of_columns * p_z_co2) };
                }
                else {
                    //new_points[i * 8 + triangle[0]] = { static_cast<int>(100 * p_x_co2),static_cast<int>(100 * p_y_co2), static_cast<int>(100 * p_z_co2) };
                }
                if (static_cast<int>(100 * p_z_co3) != 0) {
                    new_points[i * 8 + triangle[2]] = { (number_of_columns * p_x_co3 / p_z_co3),(number_of_columns * p_y_co3 / p_z_co3), (number_of_columns * p_z_co3) };
                }
                else {
                    //new_points[i * 8 + triangle[0]] = { static_cast<int>(100 * p_x_co3),static_cast<int>(100 * p_y_co3), static_cast<int>(100 * p_z_co3) };
                }
                //new_points[i * 8 + triangle[0]] = { static_cast<int>(100 * p_x_co1 / p_z_co1),static_cast<int>(100 * p_y_co1 / p_z_co1), static_cast<int>(100 * p_z_co1) };
                //new_points[i * 8 + triangle[1]] = { static_cast<int>(100 * p_x_co2/p_z_co2),static_cast<int>(100 * p_y_co2 / p_z_co2), static_cast<int>(100* p_z_co2) };
                //new_points[i * 8 + triangle[2]] = { static_cast<int>(100 * p_x_co3 / p_z_co3),static_cast<int>(100 * p_y_co3 / p_z_co3), static_cast<int>(100* p_z_co3) };
            }
            else {
       //         std::cout << "e" << std::endl;
            }
        }
    }
    //std::cout << new_points.size()<<" points" << std::endl;
    if (new_points.size() != 0) {
        for (int i = 0; i < new_points.size() / 8; i++) {
            int loop_index = 0;
            for (std::vector<int> triangle : faces) {
                //std::cout << "p1 " << new_points[8 * i + triangle[0]].x <<" "<< new_points[8 * i + triangle[0]].y <<" "<< new_points[8 * i + triangle[0]].z << std::endl;
                //std::cout << "p2 " << new_points[8 * i + triangle[1]].x << " " << new_points[8 * i + triangle[1]].y << " " << new_points[8 * i + triangle[1]].z << std::endl;
                //std::cout << "p3 " << new_points[8 * i + triangle[2]].x << " " << new_points[8 * i + triangle[2]].y << " " << new_points[8 * i + triangle[2]].z << std::endl;
                    std::vector<std::vector<float>> rasterized_points = rasterize(new_points[8*i+triangle[0]], new_points[8*i+triangle[1]], new_points[8*i+triangle[2]],0);
                    //std::cout << rasterized_points.size() << std::endl;;
                    int triangle_index = 0;
                    //std::cout << rasterized_points.size() << " ";
                    for (std::vector<float> p : rasterized_points) {
                        
                        int p_x_co = p[0];
                        int p_y_co = p[1];
                        int p_z_co = p[2];
                        //std::cout << p_x_co << " " << p_y_co << std::endl;
                        if (p_z_co != 0 && abs(1 * p_y_co)<number_of_columns/2 && abs(1 * p_x_co )<characters_per_row/2) {
                            if (p_z_co >0&&p_z_co < screen[characters_per_row * floor(1 * p_y_co ) + number_of_columns/2 * characters_per_row + characters_per_row/2 + 1 * p_x_co][1]) {
                                //std::cout << "giuhwrguiw" << std::endl;
                                screen[characters_per_row * floor(1 * p_y_co) + number_of_columns / 2 * characters_per_row + characters_per_row / 2 + 1 * p_x_co] = { triangle[3],p_z_co};
                            }
                        }
                    }
                    loop_index++;
            }
        }
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(000));
}

void controls(float& x_rotation, float& y_rotaion, float& px, float& py, float& pz) {
    if (GetAsyncKeyState(VK_UP) & 0x8000) {
        y_rotation += 0.1;
    }
    if (GetAsyncKeyState(VK_DOWN) & 0x8000) {
         y_rotation -= 0.1;
    }
    if (GetAsyncKeyState(VK_LEFT) & 0x8000) {
        x_rotation -= 0.1;
    }
    if (GetAsyncKeyState(VK_RIGHT) & 0x8000) {
        x_rotation += 0.1;
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
    std::vector<std::vector<int>> screen(characters_per_row * number_of_columns);
    cuboid_to_vertices(points);
    // Seed random number generator
    srand(static_cast<unsigned int>(time(nullptr)));

    while (true) {
        controls(x_rotation, y_rotation,px,py,pz);
        update_screen(screen, points, x_rotation, y_rotation, px, py, pz); // Update screen content
        //std::cout << "update" << std::endl;
        draw_screen(screen);   // Draw updated screen
        //std::cout << "draw" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}

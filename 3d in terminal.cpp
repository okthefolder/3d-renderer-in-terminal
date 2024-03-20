#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>
#include <cmath>
#include <Windows.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <unordered_map>
#include <tuple>
#include <mutex>
#include <omp.h>
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

struct BoundingBox {
    float px;
    float py;
    float pz;
    float size; // Size of the bounding box (assuming it's a cube)
};

struct Point2 {
    float x;
    float y;
    float z;

};

struct TupleHash {
    template <typename T, typename U, typename V>
    std::size_t operator()(const std::tuple<T, U, V>& t) const {
        auto hash1 = std::hash<T>{}(std::get<0>(t));
        auto hash2 = std::hash<U>{}(std::get<1>(t));
        auto hash3 = std::hash<V>{}(std::get<2>(t));
        return hash1 ^ hash2 ^ hash3;
    }
};

struct TupleEqual {
    template <typename T, typename U, typename V>
    bool operator()(const std::tuple<T, U, V>& t1, const std::tuple<T, U, V>& t2) const {
        return t1 == t2;
    }
};

const int world_x = 16;
const int world_y = 16;
const int world_z = 16;
int unique_blocks = 1;
//for 1 (1880,480)
//for 12 (156,40)
//for 7 (268,68)
// for 5 (313, 80)
//keep the numbers even
const int characters_per_row = 940;
const int number_of_columns = 230;
float dy = 0;

std::vector<char> characters = {
        '.', '-', ':', '_', ',', '!', 'r', 'c', 'z', 's', 'L', 'T', 'v', ')', 'J', '7', '(', 'F', 'i', '{', 'C', '}', 'f', 'I', '3', '1', 't', 'l', 'u', '[', 'n', 'e', 'o', 'Z', '5', 'Y', 'x', 'j', 'y', 'a', ']', '2', 'E', 'S', 'w', 'q', 'k', 'P', '6', 'h', '9', 'd', '4', 'V', 'p', 'O', 'G', 'b', 'U', 'A', 'K', 'X', 'H', 'm', '8', 'R', 'D', '#', '$', 'B', 'g', '0', 'M', 'N', 'W', 'Q', '@'
};

std::vector<std::vector<int>> points = {
   // {0,0,0,1,1,1},
    //{1,0,0,2,1,1}
};


float x_rotation = 0;
float y_rotation = 0;
float px = 1024 * 1024;
float py = 8*16+32;
float pz = 1024*1024;

uint64_t hashCoordinates(int x, int y) {
    // Combine the two integers into a single value using bitwise operations
    uint64_t combinedValue = static_cast<size_t>(x) | (static_cast<size_t>(y) << 16);
    return combinedValue;
}

std::vector<float> getDeerministicRandomVector(int x, int y) {
    // Use a hash function to generate a deterministic seed
    // You can use any hash function that suits your needs
    uint64_t seed = hashCoordinates(x, y);

    // Create a random number generator with the deterministic seed
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0, 2 * 3.14159); // Uniform distribution from 0 to 2π

    // Generate a random angle
    float angle = dis(gen);

    // Convert the angle to a 2D unit vector
    float vx = cos(angle);
    float vy = sin(angle);

    return { vx, vy };
}

float dot_product(std::vector<float> v1, std::vector<float> v2) {
    return v1[0] * v2[0] + v1[1] * v2[1];
}

float bilinear_interpolation(float v00, float v01, float v10, float v11, float dx, float dy) {
    // Interpolate along x-axis (lerp)
    float interpolated_x0 = v00 + dx * (v10 - v00);
    float interpolated_x1 = v01 + dx * (v11 - v01);

    // Interpolate along y-axis (lerp)
    return interpolated_x0 + dy * (interpolated_x1 - interpolated_x0);
}

float perlin_noise_at_point(float fx, float fy, std::vector<std::vector<float>> perlin_noise) {
    int x0 = floor(fx);
    int y0 = floor(fy);
    float dx = fx - x0;
    float dy = fy - y0;

    // Get noise values at the four corners of the cell
    float v00 = perlin_noise[x0][y0];
    float v01 = perlin_noise[x0][y0 + 1];
    float v10 = perlin_noise[x0 + 1][y0];
    float v11 = perlin_noise[x0 + 1][y0 + 1];

    // Perform bilinear interpolation
    return bilinear_interpolation(v00, v01, v10, v11, dx, dy);
}

std::vector<std::vector<float>> perlin_noise_generation(float fx, float fy) {
    std::vector<std::vector<std::vector<float>>> random_vectors(16, std::vector<std::vector<float>>(16, std::vector<float>(2)));
    std::vector<std::vector<float>> non_interpoluated_perlin_noise(16, std::vector<float>(16, 0));
    for (int x = 0; x < 16; ++x) {
        for (int y = 0; y < 16; y++) {
            random_vectors[x][y] = getDeerministicRandomVector(x, y);
            std::cout << random_vectors[x][y][0] << " " << random_vectors[x][y][1] << "   " << std::endl;
        }
        std::cout << " " << std::endl;
    }
    std::cout << " " << std::endl;
    std::cout << " " << std::endl;
    std::cout << " " << std::endl;

    for (int x = 0; x < 15; ++x) {
        for (int y = 0; y < 15; y++) {
            non_interpoluated_perlin_noise[x][y] += dot_product(random_vectors[x][y], { fx,fy });
        }
    }
    for (int x = 0; x < 15; ++x) {
        for (int y = 0; y < 15; y++) {
            non_interpoluated_perlin_noise[x][y] += dot_product(random_vectors[x + 1][y], { 1 - fx,fy });
        }
    }
    for (int x = 0; x < 15; ++x) {
        for (int y = 0; y < 15; y++) {
            non_interpoluated_perlin_noise[x][y] += dot_product(random_vectors[x][y + 1], { fx,1 - fy });
        }
    }
    for (int x = 0; x < 15; ++x) {
        for (int y = 0; y < 15; y++) {
            non_interpoluated_perlin_noise[x][y] += dot_product(random_vectors[x + 1][y + 1], { 1 - fx,1 - fy });
        }
    }

    for (int x = 0; x < 16; ++x) {
        for (int y = 0; y < 16; y++) {
            non_interpoluated_perlin_noise[x][y] /= 4;
            std::cout << non_interpoluated_perlin_noise[x][y] << " ";
        }
        std::cout << " " << std::endl;
    }


    std::vector<std::vector<float>> perlin_noise(256, std::vector<float>(256, 0));
    for (int x = 0; x < 256-16; x++) {
        for (int y = 0; y < 256 - 16; y++) {
            perlin_noise[x][y] = perlin_noise_at_point(16 * static_cast<float>(x) / 256, 16 * static_cast<float>(y) / 256, non_interpoluated_perlin_noise);
        }
    }
    return perlin_noise;

}

std::vector<std::vector<float>> perlin_noise_at_chunk(int cx, int cy, int cz, float fx, float fz) {
    std::vector<std::vector<std::vector<float>>> random_vectors(3, std::vector<std::vector<float>>(3, std::vector<float>(2)));
    // Initialize random vectors
    for (int x = 0; x < 3; ++x) {
        for (int z = 0; z < 3; ++z) {
            random_vectors[x][z] = getDeerministicRandomVector(x + cx, z + cz);
        }
    }

    // Initialize non-interpolated perlin noise
    std::vector<std::vector<float>> non_interpolated_perlin_noise(2, std::vector<float>(2, 0));

    // Calculate perlin noise
    for (int x = 0; x < 2; ++x) {
        for (int z = 0; z < 2; ++z) {
            non_interpolated_perlin_noise[x][z] += dot_product(random_vectors[x][z], { fx, fz });
            non_interpolated_perlin_noise[x][z] += dot_product(random_vectors[x + 1][z], { 1 - fx, fz });
            non_interpolated_perlin_noise[x][z] += dot_product(random_vectors[x][z + 1], { fx, 1 - fz });
            non_interpolated_perlin_noise[x][z] += dot_product(random_vectors[x + 1][z + 1], { 1 - fx, 1 - fz });
            non_interpolated_perlin_noise[x][z] /= 4;
        }
    }

    // Interpolate perlin noise for each point
    std::vector<std::vector<float>> perlin_noise(16, std::vector<float>(16, 0));
    for (int x = 0; x < 16; ++x) {
        for (int z = 0; z < 16; ++z) {
            perlin_noise[x][z] = perlin_noise_at_point(static_cast<float>(x) / 16, static_cast<float>(z) / 16, non_interpolated_perlin_noise);
        }
    }
    return perlin_noise;
}

std::vector<uint64_t> make_chunk(int cx, int cy, int cz) {
    std::vector<std::vector<float>> perlin_noise = perlin_noise_at_chunk(cx, cy, cz, 0.5, 0.5);
    std::vector<uint64_t> chunk(16*16,0);
    for (int x = 0; x < 16; x++) {
        for (int z=0; z < 16; z++) {
            int num1 = x;
            //int num2 = 0;
            int num2 = 8*16-16*cy+(perlin_noise[x][z]+1) * 8;
            //if (perlin_noise[x][z] > 1 || perlin_noise[x][z] < -1) {
            //    std::cout << "esogs" << std::endl;
            //}
            int num3 = z;
            //std::cout << num2 << std::endl;
            for (int i = 0; i < 1; i++) {
                if (num2 + i < 16 && num2 + i >= 0) {
                    chunk[16 * num3 + num2+i] |= (static_cast<uint64_t>(0b1111) << (4 * x));
                    //std::cout << chunk[16 * num3 + num2 + i] << std::endl;
                }
            }
        }
    }
    return chunk;
}

void prepare_points(std::vector<std::vector<int>>& points) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define the range [0.0f, 1.0f) for the distribution (float)
    std::uniform_int_distribution<int> dist(-100, 100);

    // Generate and print 3 random floating-point numbers (float)
    std::vector<std::vector<float>> perlin_noise = perlin_noise_generation(0.5, 0.5);
    std::vector<std::vector<int>> chunk;
    points.reserve(128*128*20+10);
    for (int x = 1; x < 16 * 16-1; ++x) {
        for (int z = 1; z < 16 * 16-1; ++z) {
            int num1 = x;
            int num2 = 8*16+perlin_noise[x][z] * 20;
            int num3 = z;
            for (int i = -1; i <20 ; i++) {
                points.push_back({ num1,num2+i,num3,num1 + 1,num2+i + 1,num3 + 1 });
            }
        }
    }
}

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
    constexpr int buffer_size = characters_per_row * number_of_columns;
    char* buffer = new char[buffer_size];

    // Fill the buffer with characters or spaces based on screen data
    for (int i = 0; i < buffer_size; ++i) {
        int value = screen[i][0];
        buffer[i] = (value != 0) ? characters[value] : ' ';
    }

    // Clear the screen before printing
    clear_screen();

    // Write the entire buffer to the output stream at once
    std::cout.write(buffer, buffer_size);
    delete[] buffer;
}

float distanceToPlane(Point3 point, float angleX, float angleY) {
    // Calculate camera's viewing direction vector
    float vx = cos(angleY) * sin(angleX);
    float vy = sin(angleY);
    float vz = cos(angleY) * cos(angleX);

    // Compute the dot product of the vector from the camera position to the point
    // with the normal vector of the camera plane (which is the camera's viewing direction)
    float dotProduct = point.x * vx + point.y * vy + point.z * vz;

    // Compute the magnitude of the camera's viewing direction vector
    float magnitude = (vx * vx + vy * vy + vz * vz);

    // Compute the signed distance between the point and the camera plane
    float distance = (dotProduct) / magnitude;

    return distance;
}

bool isPointInFrontOfCamera(float angleX, float angleY, Point3 point) {
    // Calculate camera's viewing direction vector
    float vx = cos(angleY) * sin(angleX);
    float vy = sin(angleY);
    float vz = cos(angleY) * cos(angleX);

    // Compute dot product
    //float dotProduct = point.x * vx + point.y * vy + point.z * vz;

    float d = distanceToPlane(point, angleX, angleY);

    // If dot product is positive, point is in front of camera
    return d>0.2;
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

std::vector<std::vector<float>> rasterize(Point2 a, Point2 b, Point2 c) {
    std::vector<std::vector<float>> rasterized;
    std::vector<float> x_co_for_lines_1;
    std::vector<float> x_co_for_lines_2;
    std::vector<float> z_co_for_lines_1;
    std::vector<float> z_co_for_lines_2;
    order_points(a, b, c);
    std::swap(c, a);
    if (max(abs(a.x), abs(b.x), abs(c.x)) >     characters_per_row / 2 || (max(abs(a.y), abs(b.y), abs(c.y)) > number_of_columns / 2)) {
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
                    points_for_triangulation.push_back({screen_vertices[i].x,screen_vertices[i].y, p_z});
                }
            }
            if (points_for_triangulation.size() != 0) {
                float centroidx = 0;
                float centroidy = 0;
                for (Point2 p : points_for_triangulation) {
                    
                    centroidx += p.x;
                    centroidy += p.y;
                }
                for (int i = 0; i < points_for_triangulation.size(); i++) {
                }
                Point2 centroid = { centroidx / (points_for_triangulation.size()),centroidy / (points_for_triangulation.size()) ,1 };
                std::sort(points_for_triangulation.begin(), points_for_triangulation.end(), [&](const Point2& p1, const Point2& p2) {
                    return compareAngles(p1, p2, centroid);
                    });
                for (int i = 1; i < points_for_triangulation.size() - 1; i++) {
                    for (std::vector<float> tri_rasteri : rasterize(points_for_triangulation[0], points_for_triangulation[i], points_for_triangulation[i + 1])) {
                        rasterized.push_back(tri_rasteri);
                    }
                }
            }
            else {
            }
            return rasterized;
    }
    rasterized.reserve(1 / 2 * (a.x * (b.y - c.y) + b.x * (c.y - a.x) + c.x * (a.y - b.y)));
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
    return rasterized;
}

std::vector<std::vector<int>> blocks_from_chunk(const std::unordered_map<std::tuple<int, int, int>, std::vector<uint64_t>, TupleHash, TupleEqual>& chunks, int cx, int cy, int cz) {
    std::unordered_map<std::tuple<int, int, int>, std::vector<uint64_t>, TupleHash, TupleEqual>::const_iterator it = chunks.find(std::make_tuple(cx, cy, cz));
    const std::vector<uint64_t>& chunk = it->second;
    std::vector<std::vector<int>> blocks;
    blocks.reserve(256 * 16); // Reserve memory for efficiency
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 16; j++) {
            if (((chunk[i] >> 4 * j) & (0b1111)) != 0) {
                int block_x = 16 * (cx) + j;
                int block_y = 16 * (cy) + i % 16;
                int block_z = 16 * (cz) + i / 16;
                blocks.emplace_back(std::vector<int>{block_x, block_y, block_z});
            }
        }
    }
    return blocks;
}

bool sort_on_midpoint(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> mid_a = { a[0] + a[3] + a[6], a[1] + a[4] + a[7], a[2] + a[5] + a[8] };
    std::vector<int> mid_b = { b[0] + b[3] + b[6], b[1] + b[4] + b[7], b[2] + b[5] + b[8] };

    // Compare midpoints
    if (mid_a[0] != mid_b[0]) return mid_a[0] < mid_b[0]; // First compare mid_a[0] and mid_b[0]
    if (mid_a[1] != mid_b[1]) return mid_a[1] < mid_b[1]; // If equal, compare mid_a[1] and mid_b[1]
    return mid_a[2] <= mid_b[2]; // If still equal, compare mid_a[2] and mid_b[2]
}

bool compare_equality_on_midpoint(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> mid_a = { a[0] + a[3] + a[6], a[1] + a[4] + a[7], a[2] + a[5] + a[8] };
    std::vector<int> mid_b = { b[0] + b[3] + b[6], b[1] + b[4] + b[7], b[2] + b[5] + b[8] };

    // Convert midpoints to 64-bit integers for comparison
    uint64_t a_compressed = (static_cast<uint64_t>(mid_a[0]) << 32) | (static_cast<uint64_t>(mid_a[1]) << 16) | static_cast<uint64_t>(mid_a[2]);
    uint64_t b_compressed = (static_cast<uint64_t>(mid_b[0]) << 32) | (static_cast<uint64_t>(mid_b[1]) << 16) | static_cast<uint64_t>(mid_b[2]);

    // Sort in ascending order
   // std::cout << a_compressed << " "<<b_compressed << std::endl;
    return ((mid_a[0] == mid_b[0]) && (mid_a[1] == mid_b[1]) && (mid_a[2] == mid_b[2]));
}

bool air_next_to(const std::unordered_map<std::tuple<int,int,int>, std::vector<uint64_t>, TupleHash, TupleEqual>& chunks, int x, int y, int z) {
    int in_cx = (16 + x % 16) % 16;
    int in_cy = (16 + y % 16) % 16;
    int in_cz = (16 + z % 16) % 16;
    int cx = (x) / 16;
    int cy = (y) / 16;
    int cz = (z) / 16;
    //std::cout << 256 * cz + 16 * cy + cx << std::endl;
    std::tuple<int, int, int> key = std::make_tuple(cx, cy, cz);
    std::unordered_map<std::tuple<int,int,int>, std::vector<uint64_t>, TupleHash, TupleEqual>::const_iterator it = chunks.find(key);
    
    if (it != chunks.end()) {
        const std::vector<uint64_t>& chunk = it->second;
        return (((chunk[16 * in_cz + in_cy] >> (4 * in_cx)) & (static_cast<uint64_t> (0b1111))) == 0);
        // Now you can work with 'chunk'
    }
    else {
        return false;
    } 
}

//chunks[256 * chunk_z + 16 * chunk_y + chunk_x][chunk_block_index] |= (static_cast<uint64_t>(block[3]) << (4 * block_chunk_x));

std::vector<std::vector<int>> chunk_to_triangles(const std::unordered_map<std::tuple<int,int,int>, std::vector<uint64_t>, TupleHash, TupleEqual>& chunks, int cx,int cy,int cz) {
    std::vector<std::vector<int>> blocks=blocks_from_chunk(chunks, cx, cy, cz);
    //std::cout << blocks.size() << std::endl;;
    //std::cout << "tytyt" << std::endl;
    std::vector<std::vector<int>> triangles;
    std::vector<std::vector<int>> faces = {
        // Front face
        {0, 2, 1,2},
        {0, 3, 2,2},

        // Left face
        {0, 7, 4,3},
        {0, 3, 7,3},

        // Bottom face
        {0, 1, 5,1},
        {0, 4, 5,1},

        // Back face
        {4, 6, 5,4},
        {4, 7, 6,4},

        // Right face
        {1, 6, 5,5},
        {1, 2, 6,5},

        // Top face
        {3, 2, 6,30},
        {3, 7, 6,30}
    };
    for (std::vector<int> block : blocks) {
       // std::cout << block[0] << " " << block[1] << " " << block[2] << std::endl;
        std::vector<std::vector<int>> vertices = make_cuboid(block[0], block[1], block[2], block[0] + 1, block[1] + 1, block[2] + 1);
        if (air_next_to(chunks, block[0] - 1, block[1], block[2])) {
            for (int i = 2; i < 4; i++) {
                std::vector<int> triangle = faces[i];
                triangles.push_back({ vertices[triangle[0]][0], vertices[triangle[0]][1], vertices[triangle[0]][2], vertices[triangle[1]][0], vertices[triangle[1]][1], vertices[triangle[1]][2], vertices[triangle[2]][0], vertices[triangle[2]][1], vertices[triangle[2]][2], triangle[3] });
            }
        }
        if (air_next_to(chunks, block[0] + 1, block[1], block[2])) {
            for (int i = 8; i < 10; i++) {
                std::vector<int> triangle = faces[i];
                triangles.push_back({ vertices[triangle[0]][0], vertices[triangle[0]][1], vertices[triangle[0]][2], vertices[triangle[1]][0], vertices[triangle[1]][1], vertices[triangle[1]][2], vertices[triangle[2]][0], vertices[triangle[2]][1], vertices[triangle[2]][2], triangle[3] });
            }
        }


        if (air_next_to(chunks, block[0], block[1]-1, block[2])) {
            for (int i = 4; i < 6; i++) {
                std::vector<int> triangle = faces[i];
                triangles.push_back({ vertices[triangle[0]][0], vertices[triangle[0]][1], vertices[triangle[0]][2], vertices[triangle[1]][0], vertices[triangle[1]][1], vertices[triangle[1]][2], vertices[triangle[2]][0], vertices[triangle[2]][1], vertices[triangle[2]][2], triangle[3] });
            }
        }
        if (air_next_to(chunks, block[0], block[1]+1, block[2])) {
            for (int i = 10; i < 12; i++) {
                std::vector<int> triangle = faces[i];
                triangles.push_back({ vertices[triangle[0]][0], vertices[triangle[0]][1], vertices[triangle[0]][2], vertices[triangle[1]][0], vertices[triangle[1]][1], vertices[triangle[1]][2], vertices[triangle[2]][0], vertices[triangle[2]][1], vertices[triangle[2]][2], triangle[3] });
            }
        }
        
        if (air_next_to(chunks, block[0], block[1], block[2]-1)) {
            for (int i = 0; i < 2; i++) {
                std::vector<int> triangle = faces[i];
                triangles.push_back({ vertices[triangle[0]][0], vertices[triangle[0]][1], vertices[triangle[0]][2], vertices[triangle[1]][0], vertices[triangle[1]][1], vertices[triangle[1]][2], vertices[triangle[2]][0], vertices[triangle[2]][1], vertices[triangle[2]][2], triangle[3] });
            }
        }
        if (air_next_to(chunks, block[0], block[1], block[2]+1)) {
            for (int i = 6; i < 8; i++) {
                std::vector<int> triangle = faces[i];
                triangles.push_back({ vertices[triangle[0]][0], vertices[triangle[0]][1], vertices[triangle[0]][2], vertices[triangle[1]][0], vertices[triangle[1]][1], vertices[triangle[1]][2], vertices[triangle[2]][0], vertices[triangle[2]][1], vertices[triangle[2]][2], triangle[3] });
            }
        }
    }
    return triangles;
}



void update_screen(std::vector<std::vector<int>>& screen, const std::unordered_map<std::tuple<int,int,int>, std::vector<std::vector<int>>, TupleHash, TupleEqual>& map_triangles, float x_rotation, float y_rotation, float px, float py, float pz) {
    //std::cout << "screen" << std::endl;
    screen.assign(characters_per_row * number_of_columns, { 0, 1000000000 });
    int u = 0;

    for (int i = 0; i < map_triangles.size(); ++i) {
        std::unordered_map<std::tuple<int, int, int>, std::vector<std::vector<int>>, TupleHash, TupleEqual>::const_iterator Mtriangles = std::next(map_triangles.begin(), i); // Advance the iterator to position i
        const std::vector<std::vector<int>>& triangles = Mtriangles->second;
#pragma omp parallel
#pragma omp for
        for (int j = 0; j < triangles.size(); ++j) {
            const std::vector<int>& triangle = triangles[j];
            //int thread_id = omp_get_thread_num();
            //printf("Thread %d executing iteration %d\n", thread_id);
            u++;
            //            std::cout << "hhuuhu" << std::endl;
            float p_x_co1 = triangle[0] - px;
            float p_y_co1 = triangle[1] - py;
            float p_z_co1 = triangle[2] - pz;

            float p_x_co2 = triangle[3] - px;
            float p_y_co2 = triangle[4] - py;
            float p_z_co2 = triangle[5] - pz;

            float p_x_co3 = triangle[6] - px;
            float p_y_co3 = triangle[7] - py;
            float p_z_co3 = triangle[8] - pz;
            Point3 point1 = { p_x_co1, p_y_co1, p_z_co1 };
            Point3 point2 = { p_x_co2, p_y_co2, p_z_co2 };
            Point3 point3 = { p_x_co3, p_y_co3, p_z_co3 };

            if (isPointInFrontOfCamera(x_rotation, y_rotation, point1) && isPointInFrontOfCamera(x_rotation, y_rotation, point2) && isPointInFrontOfCamera(x_rotation, y_rotation, point3)) {
                //std::cout << p_z_co1 << std::endl;
                //std::cout << "p" << std::endl;
                add_rotation(x_rotation, y_rotation - 3.14 / 2, p_x_co1, p_y_co1, p_z_co1);
                add_rotation(x_rotation, y_rotation - 3.14 / 2, p_x_co2, p_y_co2, p_z_co2);
                add_rotation(x_rotation, y_rotation - 3.14 / 2, p_x_co3, p_y_co3, p_z_co3);

                //project on screen
                p_x_co1 = number_of_columns * p_x_co1 / p_z_co1;
                p_y_co1 = number_of_columns * p_y_co1 / p_z_co1;
                p_z_co1 = number_of_columns * p_z_co1;

                p_x_co2 = number_of_columns * p_x_co2 / p_z_co2;
                p_y_co2 = number_of_columns * p_y_co2 / p_z_co2;
                p_z_co2 = number_of_columns * p_z_co2;

                p_x_co3 = number_of_columns * p_x_co3 / p_z_co3;
                p_y_co3 = number_of_columns * p_y_co3 / p_z_co3;
                p_z_co3 = number_of_columns * p_z_co3;

                //rasterize the triangle
                std::vector<std::vector<float>> rasterized_points = rasterize({ p_x_co1,p_y_co1 ,p_z_co1 }, { p_x_co2,p_y_co2 ,p_z_co2 }, { p_x_co3,p_y_co3 ,p_z_co3 });
                //std::cout << rasterized_points.size() << std::endl;
                for (const std::vector<float>& p : rasterized_points) {

                    int p_x_co = p[0];
                    int p_y_co = p[1];
                    int p_z_co = p[2];
                    //std::cout << p_x_co << " " << p_y_co << std::endl;
                    if (p_z_co != 0 && abs(1 * p_y_co) < number_of_columns / 2 && abs(1 * p_x_co) < characters_per_row / 2) {
                        if (p_z_co > 0 && p_z_co < screen[characters_per_row * floor(1 * p_y_co) + number_of_columns / 2 * characters_per_row + characters_per_row / 2 + 1 * p_x_co][1]) {
                            //std::cout << "giuhwrguiw" << std::endl;
                            #pragma omp critical
                            {
                                screen[characters_per_row * floor(1 * p_y_co) + number_of_columns / 2 * characters_per_row + characters_per_row / 2 + 1 * p_x_co] = { triangle[9],p_z_co };
                            }
                        }
                    }
                }
            }


        }
        
    }
}

void collisions(float px, float py, float pz, float& n_px, float& n_py, float& n_pz, const std::vector<std::vector<int>>& blocks) {
    BoundingBox newPlayer = { px + n_px, py + n_py, pz + n_pz, 0.5 };
    BoundingBox newPlayer2 = { px, py, pz, 0.5 };
    for (std::vector<int> block : blocks) {
        float bx = block[0];
        float by = block[1];
        float bz = block[2];

        if ((newPlayer.px - newPlayer.size < bx + 1 && newPlayer.px + newPlayer.size > bx) &&
            (newPlayer.py - 2*newPlayer.size < by + 1 && newPlayer.py + 2*newPlayer.size > by) &&
            (newPlayer.pz - newPlayer.size < bz + 1 && newPlayer.pz + newPlayer.size > bz)) {
            if ((newPlayer.px - n_px - newPlayer.size < bx + 1 && newPlayer.px - n_px + newPlayer.size > bx) &&
                (newPlayer.py -2* newPlayer.size < by + 1 && newPlayer.py + 2*newPlayer.size > by) &&
                (newPlayer.pz - n_pz - newPlayer.size < bz + 1 && newPlayer.pz - n_pz + newPlayer.size > bz)) {
                n_py = 0;
            }

            if ((newPlayer.px - newPlayer.size < bx + 1 && newPlayer.px + newPlayer.size > bx) &&
                (newPlayer.py - n_py - 2 * newPlayer.size < by + 1 && newPlayer.py - n_py + 2 * newPlayer.size > by) &&
                (newPlayer.pz - n_pz - newPlayer.size < bz + 1 && newPlayer.pz - n_pz + newPlayer.size > bz)) {
                n_px = 0;
            }

            if ((newPlayer.px - n_px - newPlayer.size < bx + 1 && newPlayer.px - n_px + newPlayer.size > bx) &&
                (newPlayer.py - n_py - 2 * newPlayer.size < by + 1 && newPlayer.py - n_py + 2 * newPlayer.size > by) &&
                (newPlayer.pz - newPlayer.size < bz + 1 && newPlayer.pz + newPlayer.size > bz)) {
                n_pz = 0;
            }
        }
    }
    
}



void controls(float& x_rotation, float& y_rotaion, float& px, float& py, float& pz, float delta_time, std::vector<std::vector<int>> blocks) {
    float n_px = 0;
    float n_py = 0;
    float n_pz = 0;
    if (GetAsyncKeyState(VK_UP) & 0x8000 && y_rotation + 1 * delta_time<3.14/2) {
        y_rotation += 1 * delta_time;
    }
    if (GetAsyncKeyState(VK_DOWN) & 0x8000 && y_rotation - 1 * delta_time > -3.14 / 2) {
        y_rotation -= 1 * delta_time;
    }
    if (GetAsyncKeyState(VK_LEFT) & 0x8000) {
        x_rotation -= 1 * delta_time;
    }
    if (GetAsyncKeyState(VK_RIGHT) & 0x8000) {
        x_rotation += 1 * delta_time;
    }

    float ty = -1;
    float txz = 0;

    

    if (dy > -10) {
        dy -= 2 * delta_time;
    }
    ty = dy*delta_time;
    collisions(px, py, pz, txz, ty, txz, blocks);
    if (ty == 0) {
        //n_py = 0.1;
        dy = 0;
    }
   // std::cout << "ty" << ty << std::endl;
   // ty = 0;
    if (GetAsyncKeyState(VK_SPACE) & 0x8000 && ty == 0) {
        dy = 2;
        //n_py += 5 * delta_time;
    }
    if (GetAsyncKeyState('C') & 0x8000) {
        //n_py -= 5 * delta_time;
    }


    n_py += dy * delta_time;
    //std::cout << "ty" << ty << std::endl;
    if (GetAsyncKeyState('W') & 0x8000) { // Move forward
        n_px += -3 * std::sin(-x_rotation) * delta_time;
        n_pz -= -3 * std::cos(x_rotation) * delta_time;
    }
    if (GetAsyncKeyState('S') & 0x8000) { // Move backward
        n_px -= -3 * std::sin(-x_rotation) * delta_time;
        n_pz += -3 * std::cos(x_rotation) * delta_time;
    }
    if (GetAsyncKeyState('A') & 0x8000) { // Turn left
        n_px += -3 * std::cos(-x_rotation) * delta_time;
        n_pz -= -3 * std::sin(x_rotation) * delta_time;
    }
    if (GetAsyncKeyState('D') & 0x8000) { // Turn right
        n_px -= -3 * std::cos(-x_rotation) * delta_time;
        n_pz += -3 * std::sin(x_rotation) * delta_time;
    }
    //n_py--;
    //std::cout << "py" << n_py << std::endl;
    collisions(px, py, pz, n_px, n_py, n_pz, blocks);
    
    px += n_px;
    py += n_py;
    pz += n_pz;
}

int hash(int x, int y, int z) {
    return x + y * world_x + z * world_x * world_y;
}

void invertChunkHash(int hash_value, int& x, int& y, int& z) {
    z = hash_value / (world_x * world_y);
    y = (hash_value % (world_x * world_y)) / world_x;
    x = (hash_value % (world_x * world_y)) % world_x;
}

std::vector<std::vector<uint64_t>> blocks_to_chuncks(const std::vector<std::vector<int>>& blocks, int chunk_size) {
    std::vector<std::vector<uint64_t>> chunks(16 * 16 * 16, std::vector<uint64_t>(256));
    for (std::vector<int> block : blocks) {
        block = { block[0],block[1],block[2],1 };
        int block_chunk_x = (16 + block[0] % 16) % 16;
        int block_chunk_y = (16 + block[1] % 16) % 16;
        int block_chunk_z = (16 + block[2] % 16) % 16;
        uint64_t chunk_block_index = block_chunk_y + 16 * block_chunk_z;
        int chunk_x = ( block[0]) / 16;
        int chunk_y = (block[1]) / 16;
        int chunk_z = (block[2]) / 16;
        chunks[256 * chunk_z + 16 * chunk_y + chunk_x][chunk_block_index] |= (static_cast<uint64_t>(block[3]) << (4 * block_chunk_x));
    }
    return chunks;
}

std::vector<std::vector<int>> blocks_from_neighboring_chunks(const std::unordered_map<std::tuple<int,int,int>, std::vector<uint64_t>, TupleHash, TupleEqual>& map_chunks, float px, float py, float pz) {
    std::vector<std::vector<int>> blocks;
    int cx = static_cast<int>(px) / 16;
    int cy = static_cast<int>(py) / 16;
    int cz = static_cast<int>(pz) / 16;
    int n_px = (16 + (static_cast<int>(floor(px)) % 16)) % 16;
    int n_py = (16 + (static_cast<int>(floor(py)) % 16)) % 16;
    int n_pz = (16 + (static_cast<int>(floor(pz)) % 16)) % 16;
    //std::cout <<"p co:" << px << " " << py << " " << pz << std::endl;
    int x = 0;
    int y = 0;
    int z = 0;
    if (n_px > 13) {
        x = 1;
    }
    if (n_px < 2) {
        x = -1;
    }
    if (n_py > 13) {
        y = 1;
    }
    if (n_py < 2) {
        y = -1;
    }
    if (n_pz > 13) {
        z = 1;
    }
    if (n_pz < 2) {
        z = -1;
    }
    std::vector<int> set_x;
    if (x != 0){
        set_x = { 0,x };
    }
    else {
        set_x = { 0 };
    }

    std::vector<int> set_y;
    if (y != 0) {
        set_y = { 0,y };
    }
    else {
        set_y = { 0 };
    }

    std::vector<int> set_z;
    if (z != 0) {
        set_z = { 0,z };
    }
    else {
        set_z = { 0 };
    }
    for (int lx : set_x) {
        for (int ly : set_y) {
            for (int lz : set_z) {
                //std::cout << "h";
                //std::cout << cx + lx << " " << cy + ly << " " << cz + lz << std::endl;
                if (map_chunks.find(std::make_tuple(cx + lx, cy + ly, cz + lz)) != map_chunks.end()) {
                    std::vector<uint64_t> chunk = map_chunks.find(std::make_tuple(cx + lx, cy + ly, cz + lz))->second;
                    //std::cout << "k" << std::endl;
                    for (int i = 0; i < 256; i++) {
                        for (int j = 0; j < 16; j++) {
                            if (((chunk[i] >> 4 * j) & (0b1111)) != 0) {
                                blocks.push_back({ 16 * (cx + lx) + j,(cy + ly) * 16 + i % 16,(cz + lz) * 16 + i / 16 });
                            }
                        }
                    }
                }
            }
        }
    }
    return blocks;
}



int main() {
    std::vector<std::vector<int>> screen(characters_per_row * number_of_columns);
    std::unordered_map<std::tuple<int, int, int>, std::vector<uint64_t>, TupleHash, TupleEqual> map_chunks;
    std::unordered_map<std::tuple<int, int, int>, std::vector<std::vector<int>>, TupleHash, TupleEqual> map_triangles;
    srand(static_cast<unsigned int>(time(nullptr)));
    auto last_time = std::chrono::steady_clock::now();
    int render_distance = 4;
    while (true) {
        for (int x = -render_distance; x <= render_distance; x++) {
            for (int y = -render_distance; y <= render_distance; y++) {
                for (int z = -render_distance; z <= render_distance; z++) {
                    std::tuple<int, int, int> key = std::make_tuple(px / 16 + x, py / 16 + y, pz / 16 + z);
                    std::unordered_map<std::tuple<int, int, int>, std::vector<uint64_t>, TupleHash, TupleEqual>::iterator it = map_chunks.find(key);
                    if (it == map_chunks.end()) {
                        map_chunks[key] = make_chunk(px / 16 + x, py / 16 + y, pz / 16 + z);
                        std::vector<uint64_t> chunk = map_chunks[key];
                        map_triangles[key] = chunk_to_triangles(map_chunks, px / 16 + x, py / 16 + y, pz / 16 + z);
                    }
                }
            }
        }
        std::vector<std::tuple<int, int, int>> keysToRemove;
        for (auto& pair : map_chunks) {
            const std::tuple<int, int, int>& co = pair.first;
            if (abs(std::get<0>(co) - static_cast<int>(px) / 16) > render_distance || abs(std::get<1>(co) - static_cast<int>(py) / 16) > render_distance || abs(std::get<2>(co) - static_cast<int>(pz) / 16) > render_distance) {
                keysToRemove.push_back(co);
            }
        }

        // Remove elements based on keysToRemove
        for (const auto& key : keysToRemove) {
            map_chunks.erase(key);
            map_triangles.erase(key);
        }
        //std::cout << "bus" << std::endl;
        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<float> delta_seconds = current_time - last_time;
        last_time = current_time;
        float delta_time = delta_seconds.count();
        std::cout << delta_time << '\n';
        controls(x_rotation, y_rotation,px,py,pz, delta_time, blocks_from_neighboring_chunks(map_chunks,px,py,pz));
        update_screen(screen, map_triangles, x_rotation, y_rotation, px, py, pz);
        draw_screen(screen);
    }

    return 0;
}

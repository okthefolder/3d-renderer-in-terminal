#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>

#define M_PI 3.14159265358979323846
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
    Point3 operator*(float scalar) const {
        return { x * scalar, y * scalar, z * scalar };
    }

    Point3 operator+(const Point3& other) const {
        return { x + other.x, y + other.y, z + other.z };
    }

    Point3 operator-(const Point3& other) const {
        return { x - other.x, y - other.y, z - other.z };
    }
    Point3 cross(const Point3& other) const {
        return {
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        };
    }
    Point3 operator-(const Point3& vec) {
        return { -vec.x, -vec.y, -vec.z };
    }
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
    float u;
    float v;

};

struct Point2_ref {
    float& x;
    float& y;
    float& z;
    float& u;
    float& v;

};

struct Plane {
    Point3 normal;
    float distance;
};

struct Frustum {
    Plane nearPlane;
    Plane farPlane;
    Plane leftPlane;
    Plane rightPlane;
    Plane topPlane;
    Plane bottomPlane;
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

struct TupleHash2 {
    template <typename T, typename U>
    std::size_t operator()(const std::tuple<T, U>& t) const {
        auto hash1 = std::hash<T>{}(std::get<0>(t));
        auto hash2 = std::hash<U>{}(std::get<1>(t));
        return hash1 ^ hash2;
        //return ((std::get<0>(t) << 4) | std::get<1>(t));
    }
};

struct TupleEqual {
    template <typename T, typename U, typename V>
    bool operator()(const std::tuple<T, U, V>& t1, const std::tuple<T, U, V>& t2) const {
        return t1 == t2;
    }
};

struct TupleEqual2 {
    template <typename T, typename U>
    bool operator()(const std::tuple<T, U>& t1, const std::tuple<T, U>& t2) const {
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
// for 2 (940, 238)
// for 3 (626,160)
//keep the numbers even
const int characters_per_row = 940;
const int number_of_columns = 240;
const float FOV = 2*M_PI / 3;  // Field of view in degrees
const float ASPECT_RATIO = static_cast<float>(characters_per_row) / static_cast<float>(number_of_columns);  // Width divided by height
float dy = 0;

std::vector<char> characters = {
        '.', '-', ':', '_', ',', '!', 'r', 'c', 'z', 's', 'L', 'T', 'v', ')', 'J', '7', '(', 'F', 'i', '{', 'C', '}', 'f', 'I', '3', '1', 't', 'l', 'u', '[', 'n', 'e', 'o', 'Z', '5', 'Y', 'x', 'j', 'y', 'a', ']', '2', 'E', 'S', 'w', 'q', 'k', 'P', '6', 'h', '9', 'd', '4', 'V', 'p', 'O', 'G', 'b', 'U', 'A', 'K', 'X', 'H', 'm', '8', 'R', 'D', '#', '$', 'B', 'g', '0', 'M', 'N', 'W', 'Q', '@'
};

std::vector<std::vector<int>> points = {
   // {0,0,0,1,1,1},
    //{1,0,0,2,1,1}
};

std::vector<Point2> screen_vertices = { {characters_per_row / 2 - 1,number_of_columns / 2 - 1},{-characters_per_row / 2 + 1,-number_of_columns / 2 + 1},{characters_per_row / 2 - 1,-number_of_columns / 2 + 1},{-characters_per_row / 2 + 1,number_of_columns / 2 + 1} };

std::vector<std::vector<int>> dirtTexture = {
    {1, 3, 2, 2, 1, 3, 2, 1},
    {2, 2, 1, 3, 1, 2, 3, 1},
    {3, 1, 2, 2, 3, 1, 1, 2},
    {2, 3, 1, 1, 2, 3, 1, 2},
    {1, 2, 3, 2, 1, 1, 2, 3},
    {3, 1, 2, 3, 1, 2, 2, 1},
    {2, 2, 3, 1, 2, 1, 3, 1},
    {1, 3, 1, 2, 3, 2, 1, 3}
};


float x_rotation = 0;
float y_rotation = 0;
float px = 1024*1024;
float py = 1024*1024+16;
float pz = 1024*1024;

std::vector<std::vector<int>> faces = {
    // Front face
    {0, 1, 2},
    {0, 3, 2},

    // Left face
    {0, 4, 7},
    {0, 3, 7},

    // Bottom face
    {0, 1, 5},
    {0, 4, 5},

    // Back face
    {4, 5, 6},
    {4, 7, 6},

    // Right face
    {1, 5, 6},
    {1, 2, 6},

    // Top face
    {3, 2, 6},
    {3, 7, 6}
};
std::vector<std::vector<int>> vertices = {
    {0,0,0},
    {1,0,0},
    {1,1,0},
    {0,1,0},
    {0,0,1},
    {1,0,1},
    {1,1,1},
    {0,1,1},
};

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
void fill_screen(int screen[characters_per_row * number_of_columns][4]) {
    for (int i = 0; i < characters_per_row * number_of_columns; i++) {
        screen[i][0] = 0;
        screen[i][1] = 1000000000;
        screen[i][2] = -1;
        screen[i][3] = -1;
    }
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
    std::vector<std::vector<float>> perlin_noise = perlin_noise_at_chunk(cx, cy, cz, 0.9, 0.5);
    std::vector<uint64_t> chunk(16*16,0);
    for (int x = 0; x < 16; x++) {
        for (int z=0; z < 16; z++) {
            int num1 = x;
            //int num2 = 0;
            int num2 = 1024 * 1024 -16*cy+(perlin_noise[x][z]+1) * 8;
            //num2 = 10 * 16;
            //if (perlin_noise[x][z] > 1 || perlin_noise[x][z] < -1) {
            //    std::cout << "esogs" << std::endl;
            //}
            int num3 = z;
            //std::cout << num2 << std::endl;
            for (int i = 0; i < 100; i++) {
                if (num2 - i < 16 && num2 - i >= 0) {
                //std::cout << "eroiern" << std::endl;
                //Sleep(5000);
                    //if (num1+cx*16 == 1024 * 1024 && num3+cz*16 == 1024 * 1024) {
                        //std::cout << "eroiern2" << std::endl;
                        //Sleep(5000);
                        chunk[16 * num3 + num2 - i] |= (static_cast<uint64_t>(0b1111) << (4 * x));
                    //}
                    //std::cout << chunk[16 * num3 + num2 + i] << std::endl;
                }
            }
        }
    }
    return chunk;
}

/*void prepare_points(std::vector<std::vector<int>>& points) {
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
}*/

void make_cuboid(std::vector<std::vector<int>>& vertices, const int& x1, const int& y1, const int& z1, const int& x2, const int& y2, const int& z2) {
    //std::vector<std::vector<int>> vertices;
    // Define the eight vertices of the cuboid
    /*vertices.push_back({x1, y1, z1}); // 0
    vertices.push_back({ x2, y1, z1 }); // 1
    vertices.push_back({ x2, y2, z1 }); // 2
    vertices.push_back({ x1, y2, z1 }); // 3
    vertices.push_back({ x1, y1, z2 }); // 4
    vertices.push_back({ x2, y1, z2 }); // 5
    vertices.push_back({ x2, y2, z2 }); // 6
    vertices.push_back({ x1, y2, z2 }); // 7*/
    vertices = { { x1, y1, z1 } ,{ x2, y1, z1 },{ x2, y2, z1 },{ x1, y2, z1 },{ x1, y1, z2 }, { x2, y1, z2 },{ x2, y2, z2 },{ x1, y2, z2 } };

    //return vertices;

}

/*void cuboid_to_vertices(std::vector<std::vector<int>>& points) {
    std::vector<std::vector<int>> vertices;
    for (std::vector<int> p : points) {
        for (std::vector<int> vertice : make_cuboid(p[0], p[1], p[2], p[3], p[4], p[5])) {
            vertices.push_back(vertice);
        }
    }
    points = vertices;
}*/

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

void draw_screen(const int screen[characters_per_row * number_of_columns][4]) {
    constexpr int buffer_size = characters_per_row * number_of_columns;
    char* buffer = new char[buffer_size];

    // Fill the buffer with characters or spaces based on screen data

    for (int i = 0; i < buffer_size; ++i) {
        //if (i % 1000 == 0) {
        //    std::cout << i << '\n';
        //}
        int value = screen[i][0];
        //std::cout << value << std::endl;
        
        if (screen[i][2] != -1) {
            //if (screen[i][2] <0 ) {
            //    std::cout << i <<" "<<screen[i][0] << " " << screen[i][1] << " " << screen[i][2] << " " << screen[i][3] << '\n';
            //}
            //if (abs(screen[i][2] / 100) > 10) {
                //std::cout << screen[i][2] / 100 << std::endl;
                //leep(10);
            //}

            //value += min(abs(screen[i][2]/100), 10);
        //}

            //std::cout << screen[i][2]<<" "<< screen[i][3] << std::endl;
            if (screen[i][2]!=0) {
                //Sleep(100);
            }
            if (screen[i][2] < 0 || screen[i][3] < 0) {
                std::cout << screen[i][2] << " " << screen[i][3] << std::endl;
            }
            value += dirtTexture[(7*screen[i][2])/1000][(7 * screen[i][3]) / 1000];
            //value += screen[i][2] / 200;
        }
        if (i % characters_per_row == characters_per_row - 1) {
            buffer[i] = '\n';
        }
        else {
            buffer[i] = (value != 0) ? characters[value] : ' ';
        }
        
    }

    // Clear the screen before printing
    clear_screen();

    // Write the entire buffer to the output stream at once
    std::cout.write(buffer, buffer_size);
    //printf(buffer);
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
    float z_1 = p1.z + t1 * (p2.z - p1.z);

    float t2 = ((number_of_columns - 1) / 2 - p1.y) / (p2.y - p1.y);
    float z_2 = p1.z + t2 * (p2.z - p1.z);

    l1_intersection = { ((-(number_of_columns - 1) / 2 - p1.y) * ((p1.x - p2.x) / (p1.y - p2.y)) + p1.x), -(number_of_columns - 1) / 2, z_1 };
    l2_intersection = { (((number_of_columns - 1) / 2 - p1.y) * ((p1.x - p2.x) / (p1.y - p2.y)) + p1.x), (number_of_columns - 1) / 2, z_2 };
    //std::cout << "l1 " << l1_intersection.x << " " << l1_intersection.y << std::endl;
}
if (p1.x != p2.x) {

    float t3 = (-(characters_per_row - 1) / 2 - p1.x) / (p2.x - p1.x);
    float z_3 = p1.z + t3 * (p2.z - p1.z);

    float t4 = ((characters_per_row - 1) / 2 - p1.x) / (p2.x - p1.x);
    float z_4 = p1.z + t4 * (p2.z - p1.z);

    l3_intersection = { (-(characters_per_row - 1) / 2),   (-(characters_per_row - 1) / 2 - p1.x) * ((p1.y - p2.y) / (p1.x - p2.x)) + p1.y,  z_3 };
    //std::cout << "l3.x: " << l3_intersection.x<<" chp: "<< << std::endl;
    l4_intersection = { (characters_per_row - 1) / 2,   ((characters_per_row - 1) / 2 - p1.x) * ((p1.y - p2.y) / (p1.x - p2.x)) + p1.y, z_4 };
    //std::cout << -(characters_per_row-1)/2<<" " << "l3:" << l3_intersection.x << " " << l3_intersection.y << std::endl;

}
if (l1_intersection.x <= max(p1.x, p2.x) && l1_intersection.x >= min(p1.x, p2.x) && l1_intersection.y <= max(p1.y, p2.y) && l1_intersection.y >= min(p1.y, p2.y) && abs(l1_intersection.x) < characters_per_row / 2 && abs(l1_intersection.y) < number_of_columns / 2 && l1_intersection.z != 0) {
    intersections.push_back(l1_intersection);
}
if (l2_intersection.x <= max(p1.x, p2.x) && l2_intersection.x >= min(p1.x, p2.x) && l2_intersection.y <= max(p1.y, p2.y) && l2_intersection.y >= min(p1.y, p2.y) && abs(l2_intersection.x) < characters_per_row / 2 && abs(l2_intersection.y) < number_of_columns / 2 && l2_intersection.z != 0) {
    intersections.push_back(l2_intersection);
}
if (l3_intersection.y <= max(p1.y, p2.y) && l3_intersection.y >= min(p1.y, p2.y) && l3_intersection.x <= max(p1.x, p2.x) && l3_intersection.x >= min(p1.x, p2.x) && abs(l3_intersection.x) < characters_per_row / 2 && abs(l3_intersection.y) < number_of_columns / 2 && l3_intersection.z != 0) {
    intersections.push_back(l3_intersection);
}



if (l4_intersection.y <= max(p1.y, p2.y) + 1 && l4_intersection.y >= min(p1.y, p2.y) - 1 && l4_intersection.x <= max(p1.x, p2.x) && l4_intersection.x >= min(p1.x, p2.x) && abs(l4_intersection.x) <= characters_per_row / 2 && abs(l4_intersection.y) < number_of_columns / 2 && l4_intersection.z != 0) {
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
    //std::cout << "rgrogsrg" << std::endl;
    float angle1 = atan2(p1.y - centroid.y, p1.x - centroid.x);
    float angle2 = atan2(p2.y - centroid.y, p2.x - centroid.x);
   // std::cout << angle1 << " " << angle2 << std::endl;
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

float area_of_3d_triangle(const Point2& a, const Point2& b, const Point2& c) {
    const float& x1 = a.x;
    const float& x2 = b.x;
    const float& x3 = c.x;
    const float& y1 = a.y;
    const float& y2 = b.y;
    const float& y3 = c.y;
    const float& z1 = a.z;
    const float& z2 = b.z;
    const float& z3 = c.z;

    float area = 0.5 * sqrt(0
        + ((x2 * y1) - (x3 * y1) - (x1 * y2) + (x3 * y2) + (x1 * y3) - (x2 * y3)) * ((x2 * y1) - (x3 * y1) - (x1 * y2) + (x3 * y2) + (x1 * y3) - (x2 * y3))
        + ((x2 * z1) - (x3 * z1) - (x1 * z2) + (x3 * z2) + (x1 * z3) - (x2 * z3)) * ((x2 * z1) - (x3 * z1) - (x1 * z2) + (x3 * z2) + (x1 * z3) - (x2 * z3))
        + ((y2 * z1) - (y3 * z1) - (y1 * z2) + (y3 * z2) + (y1 * z3) - (y2 * z3)) * ((y2 * z1) - (y3 * z1) - (y1 * z2) + (y3 * z2) + (y1 * z3) - (y2 * z3))
    );
    return area;
}

std::tuple<float, float> to_UV(const Point2& a, const Point2& b, const Point2& c, const float& x, const float& y, const float& z) {
    Point2 p = { x,y,z };
    const float area_abc = area_of_3d_triangle(a, b, c);
    if (area_abc < 1) {
        return std::make_tuple(0, 0);
    }


    const float u = area_of_3d_triangle(b, c, p) / area_abc;
    const float v = area_of_3d_triangle(a, c, p) / area_abc;
    const float w = 1 - u - v;
    //float uva = max(0.0f, min(1.0f, z*(u * a.u/a.z + v * b.u/b.z + w * c.u/c.z)));
    //float uvb = max(0.0f, min(1.0f, z*(u * a.v/a.z + v * b.v/b.z + w * c.v/c.z)));
    const float uva = max(0.0f,min(1.0f,(u * a.u + v * b.u + w * c.u)));
    const float uvb = max(0.0f,min(1.0f,(u * a.v + v * b.v + w * c.v)));
    //std::cout << (u * a.u + v * b.u + w * c.u) << " " << (u * a.v + v * b.v + w * c.v) << std::endl;
    //std::cout << a.u << " " << b.u << " " << c.u << " " << a.v << " " << b.v << " " << c.v << std::endl;
    //Sleep(10);
    return std::make_tuple( uva, uvb );
}

bool isPointInsideTriangle(const Point2& a, const Point2& b, const Point2& c, const Point2& p) {
    // Calculate signed area of triangle ABC
    float area_abc = ((b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y));

    // Determine the orientation of the triangle
    bool clockwise = area_abc < 0; // Clockwise if area is negative

    // Calculate barycentric coordinates
    float u = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) / area_abc;
    float v = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / area_abc;
    float w = 1.0f - u - v;

    // Check if point is inside triangle based on orientation
    if (clockwise) {
        return (u <= 0 && v <= 0 && w <= 0); // Inside if all barycentric coordinates are non-positive
    }
    else {
        return (u >= 0 && v >= 0 && w >= 0); // Inside if all barycentric coordinates are non-negative
    }
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

int air_next_to(const std::unordered_map<std::tuple<int,int,int>, std::vector<uint64_t>, TupleHash, TupleEqual>& chunks, int x, int y, int z) {
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
        return (((chunk[16 * in_cz + in_cy] >> (4 * in_cx)) & (static_cast<uint64_t> (0b1111))));
        // Now you can work with 'chunk'
    }
    else {
        return 0;
    } 
}

std::vector<int> triangle_decoder(int encoded_triangle) {
    int triangle_index = (encoded_triangle >> 12) & (0b1111);
    /*for (int i = 0; i < 12; i++) {
        std::vector<int> triangle_from_vertices = faces[i];
        std::vector<int> vertex1 = vertices[triangle_from_vertices[0]];
        std::vector<int> vertex2 = vertices[triangle_from_vertices[1]];
        std::vector<int> vertex3 = vertices[triangle_from_vertices[2]];
        std::cout << vertex1[0] << vertex1[1] << vertex1[2] << vertex2[0] << vertex2[1] << vertex2[2] << vertex3[0] << vertex3[1] << vertex3[2] << std::endl;
    }
    Sleep(100000);*/
    
    std::vector<int> vertex_data = {
        0b000100110,
        0b000010110,
        0b000001011,
        0b000010011,
        0b000100101,
        0b000001101,
        0b001101111,
        0b001011111,
        0b100101111,
        0b100110111,
        0b010110111,
        0b010011111
    };

    std::vector<int> triangle_from_vertices = faces[triangle_index];
    std::vector<int> vertex1 = vertices[triangle_from_vertices[0]];
    std::vector<int> vertex2 = vertices[triangle_from_vertices[1]];
    std::vector<int> vertex3 = vertices[triangle_from_vertices[2]];
    vertex1[0] = (vertex_data[triangle_index] >> 0) & 1;
    vertex1[1] = (vertex_data[triangle_index] >> 2) & 1;
    vertex1[2] = (vertex_data[triangle_index] >> 3) & 1;
    vertex1[3] = (vertex_data[triangle_index] >> 3) & 1;
    vertex1[4] = (vertex_data[triangle_index] >> 4) & 1;
    vertex1[5] = (vertex_data[triangle_index] >> 5) & 1;
    vertex1[6] = (vertex_data[triangle_index] >> 6) & 1;
    vertex1[7] = (vertex_data[triangle_index] >> 7) & 1;
    vertex1[8] = (vertex_data[triangle_index] >> 8) & 1;
    int cube_x = (encoded_triangle>>0) & 0b1111;
    int cube_y = (encoded_triangle>>4) & 0b1111;
    int cube_z = (encoded_triangle>>8) & 0b1111;

    //std::vector<std::vector<int>> UV_vertices = { {0,0},{1,0}, {0,1}, {1,1} };
    int UV1=0;
    int UV2=0;
    int UV3=0;
    int UV4=0;
    int UV5=0;
    int UV6=0;
    if (triangle_index % 2 == 0) {
        UV1 = 0;
        UV2 = 0;
        UV3 = 1;
        UV4 = 0;
        UV5 = 1;
        UV6 = 1;
    }
    else {
        UV1 = 0;
        UV2 = 0;
        UV3 = 0;
        UV4 = 1;
        UV5 = 1;
        UV6 = 1;
    }

    std::vector<int> triangle = {
        cube_x + vertex1[0],cube_y + vertex1[1],cube_z + vertex1[2],
        cube_x + vertex2[0],cube_y + vertex2[1],cube_z + vertex2[2],
        cube_x + vertex3[0],cube_y + vertex3[1],cube_z + vertex3[2],
        20 + 2 * (triangle_index / 2),
        UV1,UV2,
        UV3,UV4,
        UV5,UV6
    };
    return triangle;
}

int triangle_encoder(int x, int y, int z, int index) {
    int encoded_triangle =
        (x & 0b1111) |
        ((y & 0b1111) << 4) |
        ((z & 0b1111) << 8) |
        ((index & 0b1111) << 12);
    return encoded_triangle;
}

//chunks[256 * chunk_z + 16 * chunk_y + chunk_x][chunk_block_index] |= (static_cast<uint64_t>(block[3]) << (4 * block_chunk_x));
void block_to_triangles(std::unordered_map<std::tuple<int, int>, int, TupleHash2, TupleEqual2>& triangles, std::vector<int> block, const std::unordered_map<std::tuple<int, int, int>, std::vector<uint64_t>, TupleHash, TupleEqual>& chunks, const std::vector<std::vector<int>>& faces, const std::vector<std::vector<int>>& UV_vertices, const std::vector<std::vector<int>>& vertices) {
    int block_type = air_next_to(chunks, block[0], block[1], block[2]);
    if (air_next_to(chunks, block[0] - 1, block[1], block[2]) == 0) {
        int UV_index = 0;
        for (int i = 2; i < 4; i++) {
            std::vector<int> triangle = faces[i];
            if (block_type == 0b1101) {
                triangle[3] -= 15;
            }

            if (block_type != 0) {
                triangles[std::make_tuple(((block[0] % 16)) | ((block[1] % 16) << 8) | ((block[2] % 16) << 16), i)] = triangle_encoder(block[0]%16, block[1]%16, block[2]%16, i);
            }
            else {
                triangles.erase(std::make_tuple(((block[0] % 16)) | ((block[1] % 16) << 8) | ((block[2] % 16) << 16), i));
            }
            UV_index++;
        }
    }
    if (air_next_to(chunks, block[0] + 1, block[1], block[2]) == 0) {
        int UV_index = 0;
        for (int i = 8; i < 10; i++) {
            std::vector<int> triangle = faces[i];
            if (block_type == 0b1101) {
                triangle[3] -= 15;
            }
            if (block_type != 0) {
                triangles[std::make_tuple(((block[0] % 16)) | ((block[1] % 16) << 8) | ((block[2] % 16) << 16), i)] = triangle_encoder(block[0] % 16, block[1] % 16, block[2] % 16, i);
            }
            else {
                triangles.erase(std::make_tuple(((block[0] % 16)) | ((block[1] % 16) << 8) | ((block[2] % 16) << 16), i));
            }            UV_index++;
        }
    }


    if (air_next_to(chunks, block[0], block[1] - 1, block[2]) == 0) {
        int UV_index = 0;
        for (int i = 4; i < 6; i++) {
            std::vector<int> triangle = faces[i];
            if (block_type == 0b1101) {
                triangle[3] -= 15;
            }
            if (block_type != 0) {
                triangles[std::make_tuple(((block[0] % 16)) | ((block[1] % 16) << 8) | ((block[2] % 16) << 16), i)] = triangle_encoder(block[0] % 16, block[1] % 16, block[2] % 16, i);
            }
            else {
                triangles.erase(std::make_tuple(((block[0] % 16)) | ((block[1] % 16) << 8) | ((block[2] % 16) << 16), i));
            }            UV_index++;
        }
    }
    if (air_next_to(chunks, block[0], block[1] + 1, block[2]) == 0) {
        int UV_index = 0;
        for (int i = 10; i < 12; i++) {
            std::vector<int> triangle = faces[i];
            if (block_type == 0b1101) {
                triangle[3] -= 15;
            }
            if (block_type != 0) {
                triangles[std::make_tuple(((block[0] % 16)) | ((block[1] % 16) << 8) | ((block[2] % 16) << 16), i)] = triangle_encoder(block[0] % 16, block[1] % 16, block[2] % 16, i);
            }
            else {
                triangles.erase(std::make_tuple(((block[0] % 16)) | ((block[1] % 16) << 8) | ((block[2] % 16) << 16), i));
            }            UV_index++;
        }
    }

    if (air_next_to(chunks, block[0], block[1], block[2] - 1) == 0) {
        int UV_index = 0;
        for (int i = 0; i < 2; i++) {
            std::vector<int> triangle = faces[i];
            if (block_type == 0b1101) {
                triangle[3] -= 15;
            }
            if (block_type != 0) {
                triangles[std::make_tuple(((block[0] % 16)) | ((block[1] % 16) << 8) | ((block[2] % 16) << 16), i)] = triangle_encoder(block[0] % 16, block[1] % 16, block[2] % 16, i);
            }
            else {
                triangles.erase(std::make_tuple(((block[0] % 16)) | ((block[1] % 16) << 8) | ((block[2] % 16) << 16), i));
            }            UV_index++;
        }
    }
    if (air_next_to(chunks, block[0], block[1], block[2] + 1) == 0) {
        int UV_index = 0;
        for (int i = 6; i < 8; i++) {
            std::vector<int> triangle = faces[i];
            if (block_type == 0b1101) {
                triangle[3] -= 15;
            }
            if (block_type != 0) {
                triangles[std::make_tuple(((block[0] % 16)) | ((block[1] % 16) << 8) | ((block[2] % 16) << 16), i)] = triangle_encoder(block[0] % 16, block[1] % 16, block[2] % 16, i);
            }
            else {
                triangles.erase(std::make_tuple(((block[0] % 16)) | ((block[1] % 16) << 8) | ((block[2] % 16) << 16), i));
            }            UV_index++;
        }
    }
}

std::unordered_map<std::tuple<int, int>, int, TupleHash2, TupleEqual2> chunk_to_triangles(const std::unordered_map<std::tuple<int,int,int>, std::vector<uint64_t>, TupleHash, TupleEqual>& chunks, int cx,int cy,int cz) {
    std::vector<std::vector<int>> blocks=blocks_from_chunk(chunks, cx, cy, cz);
    //std::cout << blocks.size() << std::endl;;
    //std::cout << "tytyt" << std::endl;
    std::unordered_map<std::tuple<int, int>, int, TupleHash2, TupleEqual2> triangles;
    std::vector<std::vector<int>> faces = {
        // Front face
        {0, 1, 2,20},
        {0, 3, 2,20},

        // Left face
        {0, 4, 7,22},
        {0, 3, 7,22},

        // Bottom face
        {0, 1, 5,24},
        {0, 4, 5,24},

        // Back face
        {4, 5, 6,26},
        {4, 7, 6,26},

        // Right face
        {1, 5, 6,28},
        {1, 2, 6,28},

        // Top face
        {3, 2, 6,30},
        {3, 7, 6,30}
    };
    std::vector<std::vector<int>> vertices = {
        {0,0,0},
        {1,0,0},
        {1,1,0},
        {0,1,0},
        {0,0,1},
        {1,0,1},
        {1,1,1},
        {0,1,1},
    };
    std::tuple<int, int, int> key = std::make_tuple(cx, cy, cz);
    std::vector<uint64_t> chunk = chunks.find(key)->second;
    std::vector<std::vector<int>> UV_vertices = { {0,0},{1,0}, {0,1}, {1,1} };
    //std::vector<std::vector<int>> vertices(8, std::vector<int>(3));
    for (std::vector<int> block : blocks) {
        block_to_triangles(triangles, block, chunks, faces, UV_vertices, vertices);
    }
    if (triangles.size() != 0) {
        //std::cout << triangles.size() << std::endl;
        //Sleep(2);
    }
    return triangles;

}

float calculate_triangle_size(const std::vector<int>& triangle) {
    // Assuming the triangle contains vertices (x, y, z) for each point
    // You can calculate the size using the cross product of two edges of the triangle
    // Here's a simple example assuming the triangle is on the xy-plane
    float x1 = triangle[0];
    float y1 = triangle[1];
    float x2 = triangle[3];
    float y2 = triangle[4];
    float x3 = triangle[6];
    float y3 = triangle[7];

    float edge1_x = x2 - x1;
    float edge1_y = y2 - y1;
    float edge2_x = x3 - x1;
    float edge2_y = y3 - y1;

    // Calculate the cross product magnitude
    return 0.5f * std::abs(edge1_x * edge2_y - edge1_y * edge2_x);
}

bool compare_triangle_size(const std::vector<int>& triangle1, const std::vector<int>& triangle2) {
    return calculate_triangle_size(triangle1) > calculate_triangle_size(triangle2);
}

bool inside_2d_frustum(float fardist, Point3 point, Point3 p_co, float x_rotation) {
    float x0 = point.x - p_co.x;
    float z0 = point.z - p_co.z;
    float y0 = 0;
    x_rotation += -M_PI / 2;
    float new_x0 = x0 * cos(x_rotation) - z0 * sin(x_rotation);
    float new_z0 = z0 * cos(x_rotation) + x0 * sin(x_rotation);
    //add_rotation(-M_PI/2, 0, x0, y0, z0);
    x0 = new_x0;
    z0 = new_z0;

    float d = fardist;
    float h = tan(FOV / 2) * d;
    //add_rotation(-x_rotation, M_PI/2, d, y, h);
    return ((x0 * h + z0 * d > 0) && (x0 * h - z0 * d > 0));
}

double calculateTriangleArea_Point2_v(const Point2& A, const Point2& B, const Point2& C) {
    // Calculate the determinant of the matrix formed by the vertices
    double area = 0.5 * std::abs(A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y));
    return area;
}

void interpolate(Point2_ref& v1, const Point2 v2, float z) {
    float t = (z - v1.z) / (v2.z - v1.z);
    v1.x = v1.x + t * (v2.x - v1.x);
    v1.y = v1.y + t * (v2.y - v1.y);
    v1.z = z;
    v1.u = v1.u + t * (v2.u - v1.u);
    v1.v = v1.v + t * (v2.v - v1.v);
}

void update_pixel(int screen[characters_per_row*number_of_columns][4], const Point2& a, const Point2& b, const Point2& c, const float& x, const float& y, const float& z, const int& triangle_index) {
    //std::cout << x << " " << y << " " << z << std::endl;
    //if (z != 0 && abs(1 * y) < number_of_columns / 2 && abs(1 * x) < characters_per_row / 2) {
        //std::cout << characters_per_row * floor(1 * p_y_co) + number_of_columns / 2 * characters_per_row + characters_per_row / 2 + 1 * p_x_co<<"\n";
        if (z > 0 && z < screen[characters_per_row * (number_of_columns/2+static_cast<int>(y)) + (characters_per_row / 2 + static_cast<int>(x))][1]) {
            //std::vector<float> UV_co = { 0,0 };
            std::tuple<float, float> UV_co = to_UV({ a.x,a.y,a.z, a.u, a.v }, { b.x,b.y,b.z, b.u, b.v }, { c.x,c.y,c.z, c.u, c.v }, x, y, z);
            int index = characters_per_row * (number_of_columns / 2 + static_cast<int>(y)) + (characters_per_row / 2 + x);
            screen[index][0] = 20 + 2 * (triangle_index / 2);
            screen[index][1] = static_cast<int>(z);
            screen[index][2] = static_cast<int>(1000 * std::get<0>(UV_co));
            screen[index][3] = static_cast<int>(1000 * std::get<1>(UV_co));
        }
    //}
}

void rasterize(int screen[characters_per_row*number_of_columns][4], Point2 a, Point2 b, Point2 c, int triangle_index) {
    //std::vector<std::vector<float>> rasterized;
    std::vector<float> x_co_for_lines_1;
    std::vector<float> x_co_for_lines_2;
    std::vector<float> z_co_for_lines_1;
    std::vector<float> z_co_for_lines_2;
    std::vector<float> v_co_for_lines_1;
    std::vector<float> v_co_for_lines_2;
    order_points(a, b, c);
    std::swap(c, a);
    if (calculateTriangleArea_Point2_v(a, b, c) <= 1) {
        return;
    }
    //std::cout <<"start " << c.y << " " << number_of_columns / 2 << " " << max(abs(a.y), max(abs(b.y), abs(c.y))) << std::endl;
    if (max(abs(a.x), max(abs(b.x), abs(c.x))) > characters_per_row / 2 || (max(abs(a.y), max(abs(b.y), abs(c.y))) > number_of_columns / 2)) {
        
        //std::cout << "rge" << std::endl;
        std::vector<Point2> a_b_intersection = intersection(a, b);
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
        for (Point2& p : a_b_intersection) {
            // std::cout << p.x << " a_b " << p.y <<" " <<p.z<< std::endl;
            points_for_triangulation.push_back(p);
        }
        for (Point2& p : a_c_intersection) {
            //std::cout << p.x << " a_c " << p.y << " " << p.z << std::endl;
            points_for_triangulation.push_back(p);
        }
        for (Point2& p : b_c_intersection) {
            //std::cout << p.x << " b_c " << p.y << " " << p.z << std::endl;
            points_for_triangulation.push_back(p);
        }
        
        for (int i = 0; i < 4; ++i) {
            if (intpoint_inside_trigon(screen_vertices[i], a, b, c)) {
                std::vector<float> plane_coefficients = plane_equation(a, b, c);
                float p_z = -(plane_coefficients[3] + plane_coefficients[0] * screen_vertices[i].x + plane_coefficients[1] * screen_vertices[i].y) / plane_coefficients[2];
                points_for_triangulation.push_back({ screen_vertices[i].x,screen_vertices[i].y, p_z });
            }
        }
        if (points_for_triangulation.size() != 0) {
            float centroidx = 0;
            float centroidy = 0;
            for (Point2& p : points_for_triangulation) {

                centroidx += p.x;
                centroidy += p.y;
            }
            for (int i = 0; i < points_for_triangulation.size(); i++) {
            }
            Point2 centroid = { centroidx / (points_for_triangulation.size()),centroidy / (points_for_triangulation.size()) ,1 };
            //std::cout << "sort1" << std::endl;
            bool halt_test=false;
            for (int i = 0; i < points_for_triangulation.size();i++) {
                for (int j = 0; j < points_for_triangulation.size(); j++) {
                    if (i != j) {
                        if (points_for_triangulation[i].y == points_for_triangulation[j].y && points_for_triangulation[i].x == points_for_triangulation[j].x) {
                            halt_test = true;
                        }
                    }
                }
            }
            if (halt_test==false) {
                std::sort(points_for_triangulation.begin(), points_for_triangulation.end(), [&](const Point2& p1, const Point2& p2) {
                    return compareAngles(p1, p2, centroid);
                    });
                //std::cout << "sort2" << std::endl;

                for (int i = 1; i < points_for_triangulation.size() - 1; i++) {
                    rasterize(screen, points_for_triangulation[0], points_for_triangulation[i], points_for_triangulation[i + 1], triangle_index);
                    //for (std::vector<float> tri_rasteri : rasterize(points_for_triangulation[0], points_for_triangulation[i], points_for_triangulation[i + 1])) {
                    //    rasterized.push_back(tri_rasteri);
                    //}
                }
            }
            
        }
        else {
        }
        //return rasterized;
    }
    else {
        //std::cout << "rrrgrtg" << std::endl;


        if (abs(a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) <= 2) {
            //return { { a.x,a.y,a.z } };
        }
        //rasterized.reserve(0.5 * abs(a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)));
        for (float i = c.y; i >= b.y; i--) {

            if (abs(b.y - c.y) > 1) {
                x_co_for_lines_2.push_back((i - c.y) * (b.x - c.x) / (b.y - c.y) + c.x);
                z_co_for_lines_2.push_back((i - c.y) * (b.z - c.z) / (b.y - c.y) + c.z);
                //v_co_for_lines_2.push_back(c.v + (c.y - i) * (b.y - c.v) / (c.y - b.y));
            }
            else {
                x_co_for_lines_2.push_back(b.x);
                z_co_for_lines_2.push_back(b.z);
                //v_co_for_lines_2.push_back(b.v);

            }
        }
        for (float i = b.y; i >= a.y; i--) {
            if (abs(b.y - a.y) > 1) {
                x_co_for_lines_2.push_back((i - b.y) * (a.x - b.x) / (a.y - b.y) + b.x);
                z_co_for_lines_2.push_back((i - b.y) * (a.z - b.z) / (a.y - b.y) + b.z);
                //v_co_for_lines_2.push_back(b.v + (b.y - i) * (a.y - b.v) / (b.y - a.y));

            }
            else {
                x_co_for_lines_2.push_back(a.x);
                z_co_for_lines_2.push_back(a.z);
                //v_co_for_lines_2.push_back(a.v);

            }
        }
        for (float i = c.y; i >= a.y; i--) {
            if (abs(c.y - a.y) > 1) {
                x_co_for_lines_1.push_back((i - c.y) * (a.x - c.x) / (a.y - c.y) + c.x);
                z_co_for_lines_1.push_back((i - c.y) * (a.z - c.z) / (a.y - c.y) + c.z);
                //v_co_for_lines_1.push_back(c.v + (c.y - i) * (a.y - c.v) / (c.y - a.y));

            }
            else {
                x_co_for_lines_1.push_back(c.x);
                z_co_for_lines_1.push_back(c.z);
                //v_co_for_lines_1.push_back(c.v);

            }
        }
        if (x_co_for_lines_1.size() == 0) {
            std::cout << x_co_for_lines_2.size() << std::endl;
        }
        for (float i = 0; i < x_co_for_lines_1.size(); i++) {
            if (x_co_for_lines_1[i] < x_co_for_lines_2[i]) {
                for (float x = x_co_for_lines_1[i]; x <= x_co_for_lines_2[i]; x++) {
                    if (x_co_for_lines_1[i] != x_co_for_lines_2[i]) {
                        if (z_co_for_lines_1[i] + (x - x_co_for_lines_1[i]) * (z_co_for_lines_1[i] - z_co_for_lines_2[i]) / (x_co_for_lines_1[i] - x_co_for_lines_2[i]) > 0.1) {
                           // std::cout << "end " << c.y << std::endl;
                            /*
                            int y = c.y - i;
                            int z = z_co_for_lines_1[i] + (x - x_co_for_lines_1[i]) * (z_co_for_lines_1[i] - z_co_for_lines_2[i]) / (x_co_for_lines_1[i] - x_co_for_lines_2[i]);
                            if (z > 0 && z < screen[characters_per_row * (number_of_columns / 2 + static_cast<int>(y)) + (characters_per_row / 2 + x)][1]) {
                                //std::vector<float> UV_co = { 0,0 };
                                std::tuple<float, float> UV_co = to_UV({ a.x,a.y,a.z, a.u, a.v }, { b.x,b.y,b.z, b.u, b.v }, { c.x,c.y,c.z, c.u, c.v }, x, y, z);
                                int index = characters_per_row * (number_of_columns / 2 + static_cast<int>(y)) + (characters_per_row / 2 + x);
                                screen[index] = { 20 + 2 * (triangle_index / 2),static_cast<int>(z), static_cast<int>(1000 * std::get<0>(UV_co)), static_cast<int>(1000 * std::get<1>(UV_co)) };
                            }
                            */

                            update_pixel(screen, a, b, c, x, c.y - i, z_co_for_lines_1[i] + (x - x_co_for_lines_1[i]) * (z_co_for_lines_1[i] - z_co_for_lines_2[i]) / (x_co_for_lines_1[i] - x_co_for_lines_2[i]), triangle_index);

                            //rasterized.push_back({ x, c.y - i, z_co_for_lines_1[i] + (x - x_co_for_lines_1[i]) * (z_co_for_lines_1[i] - z_co_for_lines_2[i]) / (x_co_for_lines_1[i] - x_co_for_lines_2[i])});
                        }
                    }
                    else if (z_co_for_lines_1[i] > 0.1) {
                        //rasterized.push_back({ x, c.y - i, z_co_for_lines_1[i] });
                    }
                }
            }
            else {
                for (float x = x_co_for_lines_2[i]; x <= x_co_for_lines_1[i]; x++) {
                    if (x_co_for_lines_1[i] != x_co_for_lines_2[i]) {
                        if (z_co_for_lines_2[i] + (x - x_co_for_lines_2[i]) * (z_co_for_lines_1[i] - z_co_for_lines_2[i]) / (x_co_for_lines_1[i] - x_co_for_lines_2[i]) > 0.1) {
                            /*int y = c.y - i;
                            int z = z_co_for_lines_2[i] + (x - x_co_for_lines_2[i]) * (z_co_for_lines_1[i] - z_co_for_lines_2[i]) / (x_co_for_lines_1[i] - x_co_for_lines_2[i]);
                            if (z > 0 && z < screen[characters_per_row * (number_of_columns / 2 + static_cast<int>(y)) + (characters_per_row / 2 + x)][1]) {
                                //std::vector<float> UV_co = { 0,0 };
                                std::tuple<float, float> UV_co = to_UV({ a.x,a.y,a.z, a.u, a.v }, { b.x,b.y,b.z, b.u, b.v }, { c.x,c.y,c.z, c.u, c.v }, x, y, z);
                                int index = characters_per_row * (number_of_columns / 2 + static_cast<int>(y)) + (characters_per_row / 2 + x);
                                screen[index] = { 20 + 2 * (triangle_index / 2),static_cast<int>(z), static_cast<int>(1000 * std::get<0>(UV_co)), static_cast<int>(1000 * std::get<1>(UV_co)) };
                            }*/
                          //  std::cout << "end " << c.y << std::endl;
                            update_pixel(screen, a, b, c, x, c.y - i, z_co_for_lines_2[i] + (x - x_co_for_lines_2[i]) * (z_co_for_lines_1[i] - z_co_for_lines_2[i]) / (x_co_for_lines_1[i] - x_co_for_lines_2[i]), triangle_index);
                            //rasterized.push_back({ x, c.y - i, z_co_for_lines_2[i] + (x - x_co_for_lines_2[i]) * (z_co_for_lines_1[i] - z_co_for_lines_2[i]) / (x_co_for_lines_1[i] - x_co_for_lines_2[i]) });
                        }
                    }
                    else {
                        //if (z_co_for_lines_1[i]>0.1)
                        //rasterized.push_back({ x, c.y - i, z_co_for_lines_1[i] });
                    }
                }
            }
        }
    }
    //if (rasterized.size() < 2) {
        //std::cout << "rgrdhhrd" << std::endl;
    //}
    //return rasterized;
}

void update_screen(int screen[characters_per_row*number_of_columns][4], const std::unordered_map<std::tuple<int, int, int>, std::unordered_map<std::tuple<int, int>, int, TupleHash2, TupleEqual2>, TupleHash, TupleEqual>& map_triangles, const float x_rotation, const float y_rotation, const float px, const float py, const float pz) {
   // screen.assign(characters_per_row * number_of_columns, { 0, 1024 * 1024 * 1024,-1,-1 });
    fill_screen(screen);
    int u = 0;
    std::vector<int> vectors;
    std::vector<std::tuple<int, int>> keys;
    std::vector<std::tuple<int, int, int>> chunk_coordinates;
    std::vector<int> vertices {
        //left
        0b000001011,
        0b000010011,

        //front
        0b000100110,
        0b000010110,

        //bottom
        0b000100101,
        0b000001101,

        //right
        0b100101111,
        0b100110111,

        //back
        0b001101111,
        0b001011111,

        //top
        0b010110111,
        0b010011111
    };

    for (std::unordered_map<std::tuple<int, int, int>, std::unordered_map<std::tuple<int, int>, int, TupleHash2, TupleEqual2>, TupleHash, TupleEqual>::const_iterator it = map_triangles.begin(); it != map_triangles.end(); ++it) {
        const std::unordered_map<std::tuple<int, int>, int, TupleHash2, TupleEqual2>& triangles = it->second;
        std::tuple<int,int,int> key = it->first;
        float x, y, z;
        std::tie(x, y, z) = key;
        x = x * 16;
        y = y * 16;
        z = z * 16;
        if (inside_2d_frustum(20, { x,y,z }, { px-16*sin(x_rotation),py,pz - 16 * cos(x_rotation) }, x_rotation) ||
            inside_2d_frustum(20, { x+16,y,z }, { px - 16 * sin(x_rotation),py,pz - 16 * cos(x_rotation) }, x_rotation) || 
            inside_2d_frustum(20, { x,y,z+16 }, { px - 16 * sin(x_rotation),py,pz - 16 * cos(x_rotation) }, x_rotation) || 
            inside_2d_frustum(20, { x+16,y,z+16 }, { px - 16 * sin(x_rotation),py,pz - 16 * cos(x_rotation) }, x_rotation)) {
            //vectors.reserve(triangles.size());
            for (const auto& entry : triangles) {
                chunk_coordinates.push_back(std::make_tuple(x, y, z));
                vectors.push_back(entry.second);
            }
        }
    }
    //int a = 1024 * 1024;
    //vectors = { {a,a,a,a + 1,a,a,a + 1,a,a + 1,10,0,0,1,0,1,1} };
    //{ block[0] + vertices[triangle[0]][0], block[1] + vertices[triangle[0]][1], block[2] + vertices[triangle[0]][2], block[0] + vertices[triangle[1]][0], block[1] + vertices[triangle[1]][1], block[2] + vertices[triangle[1]][2], block[0] + vertices[triangle[2]][0], block[1] + vertices[triangle[2]][1], block[2] + vertices[triangle[2]][2], triangle[3], UV_vertices[0][0], UV_vertices[0][1],  UV_vertices[1 + UV_index][0],  UV_vertices[1 + UV_index][1],  UV_vertices[3][0],  UV_vertices[3][1] };

    //std::sort(vectors.begin(), vectors.end(), compare_triangle_size);
    int number_of_threads = 1;
    //const Point3 player_vector = { sin(x_rotation), tan(y_rotation), cos(x_rotation) };
    //std::cout << player_vector.x << " " << player_vector.y << " " << player_vector.z << std::endl;
    //Sleep(10000);
    std::cout << vectors.size() << std::endl;
//#pragma omp parallel for
        for (int k = 0; k < number_of_threads; k++) {
            int thread_id = omp_get_thread_num();
            auto start_time = std::chrono::steady_clock::now();
            for (int j = 0; j < vectors.size() / number_of_threads; ++j) {
                //std::cout << j << std::endl;

                float x, y, z;
                std::tie(x, y, z) = chunk_coordinates[j*number_of_threads+k];
                const int& triangle2 = vectors[j * number_of_threads + k];
                //std::cout << "bää" << std::endl;
                /*std::vector<int> triangle = triangle_decoder(triangle2);
                
                for (int i = 0; i < 3; i++) {
                    triangle[3 * i+0] += x;
                    triangle[3 * i+1] += y;
                    triangle[3 * i+2] += z;

                }*/
                Point3 triangle_normal = { 0,0,0 };
                switch ((triangle2 >> 12) & 0b1111) {
                    case 0:
                        triangle_normal = {0,0,-1};
                        break;
                    case 1:
                        triangle_normal = {0,0,-1};
                        break;
                    case 2:
                        triangle_normal = { -1,0,0 };
                        break;
                    case 3:
                        triangle_normal = { -1,0,0 };
                        break;
                    case 4:
                        triangle_normal = {0,-1,0};
                        break;
                    case 5:
                        triangle_normal = {0,-1,0};
                        break;
                    case 6:
                        triangle_normal = { 0,0,1 };
                        break;
                    case 7:
                        triangle_normal = { 0,0,1 };
                        break;
                    case 8:
                        triangle_normal = { 1,0,0 };
                        break;
                    case 9:
                        triangle_normal = { 1,0,0 };
                        break;
                    case 10:
                        triangle_normal = { 0,1,0 };
                        break;
                    case 11:
                        triangle_normal = { 0,1,0 };
                        break;

                }
                int block_x = (triangle2 >> 0) & 0b1111;
                int block_y = (triangle2 >> 4) & 0b1111;
                int block_z = (triangle2 >> 8) & 0b1111;
                int vertex_data = vertices[(triangle2 >> 12) & 0b1111];
                int UV_data=0;
                if (((triangle2 >> 12) & 0b1111) % 2 == 0) {
                    UV_data = 0b000111;
                }
                else {
                    UV_data = 0b001011;
                }
                Point3 triangle_co = {
                    block_x + x + ((vertex_data >> 0) & 0b1 + (vertex_data >> 3) & 0b1 + (vertex_data >> 6) & 0b1) / 3,
                    block_y + y + ((vertex_data >> 1) & 0b1 + (vertex_data >> 4) & 0b1 + (vertex_data >> 7) & 0b1) / 3,
                    block_z + z + ((vertex_data >> 2) & 0b1 + (vertex_data >> 5) & 0b1 + (vertex_data >> 8) & 0b1) / 3
                };
                //Point3 triangle_co = { (triangle[0] + triangle[3] + triangle[6]) / 3,(triangle[1] + triangle[4] + triangle[7]) / 3 ,(triangle[2] + triangle[5] + triangle[8]) / 3 };
                Point3 vector = { triangle_co.x - px, triangle_co.y - py, triangle_co.z - pz };
                if ((vector.x * triangle_normal.x + vector.y * triangle_normal.y + vector.z * triangle_normal.z) <= 0.1 || true) {


                    //int thread_id = omp_get_thread_num();
                    //printf("Thread %d executing iteration %d\n", thread_id);
                    u++;
                    //            std::cout << "hhuuhu" << std::endl;
                    float p_x_co1 = block_x + x + (vertex_data >> 0 & 0b1) - px;
                    float p_y_co1 = block_y + y + (vertex_data >> 1 & 0b1) - py;
                    float p_z_co1 = block_z + z + (vertex_data >> 2 & 0b1) - pz;
                    float p_u_co1 = (UV_data >> 0) & 0b1;
                    float p_v_co1 = (UV_data >> 1) & 0b1;

                    float p_x_co2 = block_x + x + (vertex_data >> 3 & 0b1) - px;
                    float p_y_co2 = block_y + y + (vertex_data >> 4 & 0b1) - py;
                    float p_z_co2 = block_z + z + (vertex_data >> 5 & 0b1) - pz;
                    float p_u_co2 = (UV_data >> 2) & 0b1;
                    float p_v_co2 = (UV_data >> 3) & 0b1;

                    float p_x_co3 = block_x + x + (vertex_data >> 6 & 0b1) - px;
                    float p_y_co3 = block_y + y + (vertex_data >> 7 & 0b1) - py;
                    float p_z_co3 = block_z + z + (vertex_data >> 8 & 0b1) - pz;
                    float p_u_co3 = (UV_data >> 4) & 0b1;
                    float p_v_co3 = (UV_data >> 5) & 0b1;


                    Point3 point1 = { p_x_co1, p_y_co1, p_z_co1 };
                    Point3 point2 = { p_x_co2, p_y_co2, p_z_co2 };
                    Point3 point3 = { p_x_co3, p_y_co3, p_z_co3 };
                    //std::cout << "böö" << std::endl;
                    if (isPointInFrontOfCamera(x_rotation, y_rotation, point1) || isPointInFrontOfCamera(x_rotation, y_rotation, point2) || isPointInFrontOfCamera(x_rotation, y_rotation, point3)) {
                        add_rotation(x_rotation, y_rotation - 3.14 / 2, p_x_co1, p_y_co1, p_z_co1);
                        add_rotation(x_rotation, y_rotation - 3.14 / 2, p_x_co2, p_y_co2, p_z_co2);
                        add_rotation(x_rotation, y_rotation - 3.14 / 2, p_x_co3, p_y_co3, p_z_co3);
                        float constant_x = number_of_columns * 4;
                        float constant_y = number_of_columns * 2;
                        float constant = 2;
                        //project on screen
                        if (abs(p_z_co1 + constant) > 0.1 && abs(p_z_co2 + constant) > 0.1 && abs(p_z_co3 + constant) > 0.1) {
                            p_x_co1 = constant_x * p_x_co1 / (p_z_co1 + constant);
                            p_y_co1 = constant_y * p_y_co1 / (p_z_co1 + constant);
                            p_z_co1 *= constant_x;

                            p_x_co2 = constant_x * p_x_co2 / (p_z_co2 + constant);
                            p_y_co2 = constant_y * p_y_co2 / (p_z_co2 + constant);
                            p_z_co2 *= constant_x;

                            p_x_co3 = constant_x * p_x_co3 / (p_z_co3 + constant);
                            p_y_co3 = constant_y * p_y_co3 / (p_z_co3 + constant);
                            p_z_co3 *= constant_x;

                            //std::vector<std::vector<float>> rasterized_points = 
                            //std::cout << "raster" << std::endl;
                            rasterize(screen, { p_x_co1,p_y_co1 ,p_z_co1, p_u_co1,p_v_co1 }, { p_x_co2,p_y_co2 ,p_z_co2, p_u_co2,p_v_co2 }, { p_x_co3,p_y_co3 ,p_z_co3, p_u_co3,p_v_co3 }, (triangle2 >> 12) & 0b1111);
                            /*for (const std::vector<float>& p : rasterized_points) {
                                float p_x_co = p[0];
                                float p_y_co = p[1];
                                float p_z_co = p[2];
                                if (p_z_co > 1024 * 1024) {
                                    std::cout << "uhriuhgruiiughriughweiugfewhif" << std::endl;
                                }
                                //std::cout << p_x_co << " " << p_y_co << std::endl;
                                if (p_z_co != 0 && abs(1 * p_y_co) < number_of_columns / 2 && abs(1 * p_x_co) < characters_per_row / 2) {
                                    //std::cout << characters_per_row * floor(1 * p_y_co) + number_of_columns / 2 * characters_per_row + characters_per_row / 2 + 1 * p_x_co<<"\n";
                                    if (p_z_co > 0 && p_z_co < screen[characters_per_row * floor(1 * p_y_co) + number_of_columns / 2 * characters_per_row + characters_per_row / 2 + 1 * p_x_co][1]) {
                                        //std::vector<float> UV_co = { 0,0 };
                                        std::tuple<float,float> UV_co = to_UV({ p_x_co1,p_y_co1,p_z_co1, p_u_co1, p_v_co1 }, { p_x_co2,p_y_co2,p_z_co2, p_u_co2, p_v_co2 }, { p_x_co3,p_y_co3,p_z_co3 , p_u_co3, p_v_co3 }, p_x_co, p_y_co, p_z_co);
                                        int index = characters_per_row * static_cast<int>(p_y_co) + number_of_columns / 2 * characters_per_row + characters_per_row / 2 + p_x_co;
                                        screen[index] = { 20+2*(((triangle2 >> 12) & 0b1111) / 2),static_cast<int>(p_z_co), static_cast<int>(1000 * std::get<0>(UV_co)), static_cast<int>(1000 * std::get<1>(UV_co)) };
                                    }
                                }
                            }*/
                        }
                    }
                }
            }
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            //CLIP TRIANGLES TO POSITIVE Z
            // //CLIP TRIANGLES TO POSITIVE Z
            // //CLIP TRIANGLES TO POSITIVE Z
            // //CLIP TRIANGLES TO POSITIVE Z
            // 
            //std::cout << "Thread " << thread_id << " took " << duration << " milliseconds" << std::endl;
        }
        if (GetAsyncKeyState('U') & 0x8000) {
            Sleep(100000);
        }
        //CLIP TRIANGLES TO POSITIVE Z
        // //CLIP TRIANGLES TO POSITIVE Z
        // //CLIP TRIANGLES TO POSITIVE Z
        // //CLIP TRIANGLES TO POSITIVE Z
        //Sleep(1000);
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
        dy -= 5 * delta_time;
    }
    if (dy < 0) {
        ty = min(-1, dy) * delta_time;
    }
    else {
        ty = max(1,dy) * delta_time;
    }
    collisions(px, py, pz, txz, ty, txz, blocks);
    if (ty == 0) {
        //std::cout << "collision" << std::endl;
        //n_py = 0.1;
        dy = 0;
    }
    else {
        //std::cout << "no no no no no" << std::endl;
    }
   // ty = 0;
   // std::cout << "ty" << ty << std::endl;
   // ty = 0;
    if (GetAsyncKeyState(VK_SPACE) & 0x8000 && ty==0) {
        dy = 3;
        //n_py += 5 * delta_time;
    }
    if (GetAsyncKeyState('C') & 0x8000) {
        //Sleep(2000);
        n_py -= 5 * delta_time;
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
    std::cout << n_py << std::endl;
    
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
                            if (((chunk[i] >> 4 * j) & (static_cast<uint64_t>(0b1111))) != 0) {
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

float distance(const std::tuple<float, float, float>& p1, const std::tuple<float, float, float>& p2) {
    float x1, y1, z1, x2, y2, z2;
    std::tie(x1, y1, z1) = p1;
    std::tie(x2, y2, z2) = p2;
    return std::sqrt((x1 - x2) * (x1 - x2) +
        (y1 - y2) * (y1 - y2) +
        (z1 - z2) * (z1 - z2));
}

// Comparator function to sort based on distance to a given point
bool compareDistance(const std::tuple<float, float, float>& p1, const std::tuple<float, float, float>& p2, const std::tuple<float, float, float>& ref) {
    return distance(p1, ref) < distance(p2, ref);
}

bool rayIntersectsAABB(const std::tuple<float, float, float>& origin,
    const std::tuple<float, float, float>& direction,
    const std::tuple<float, float, float>& lowBound,
    const std::tuple<float, float, float>& highBound) {
    float tClose = -std::numeric_limits<float>::infinity();
    float tFar = std::numeric_limits<float>::infinity();

    float invDir = 1.0f / std::get<0>(direction);

    float t1 = (std::get<0>(lowBound) - std::get<0>(origin)) * invDir;
    float t2 = (std::get<0>(highBound) - std::get<0>(origin)) * invDir;

    float tLow = min(t1, t2);
    float tHigh = max(t1, t2);

    tClose = max(tClose, tLow);
    tFar = min(tFar, tHigh);

    invDir = 1.0f / std::get<1>(direction);

    t1 = (std::get<1>(lowBound) - std::get<1>(origin)) * invDir;
    t2 = (std::get<1>(highBound) - std::get<1>(origin)) * invDir;

    tLow = min(t1, t2);
    tHigh = max(t1, t2);

    tClose = max(tClose, tLow);
    tFar = min(tFar, tHigh);

    invDir = 1.0f / std::get<2>(direction);

    t1 = (std::get<2>(lowBound) - std::get<2>(origin)) * invDir;
    t2 = (std::get<2>(highBound) - std::get<2>(origin)) * invDir;

    tLow = min(t1, t2);
    tHigh = max(t1, t2);

    tClose = max(tClose, tLow);
    tFar = min(tFar, tHigh);

    // Check if the ray intersects the AABB
    if (tClose <= tFar) {
        // Intersection point along the ray
        return true;
    }

    return false;
}bool rayIntersectsCube(const std::tuple<float, float, float>& origin, const std::tuple<float, float, float>& direction, const std::tuple<float, float, float>& p1, const std::tuple<float, float, float>& p2) {
    const std::tuple<float, float, float> minPoint = std::make_tuple(min(std::get<0>(p1), std::get<0>(p2)), min(std::get<1>(p1), std::get<1>(p2)), min(std::get<2>(p1), std::get<2>(p2)));
    const std::tuple<float, float, float> maxPoint = std::make_tuple(max(std::get<0>(p1), std::get<0>(p2)), max(std::get<1>(p1), std::get<1>(p2)), max(std::get<2>(p1), std::get<2>(p2)));
    float tmin = -std::numeric_limits<float>::infinity();
    float tmax = std::numeric_limits<float>::infinity();

    float t0 = (std::get<0>(minPoint) - std::get<0>(origin)) / std::get<0>(direction);
    float t1 = (std::get<0>(maxPoint) - std::get<0>(origin)) / std::get<0>(direction);
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    tmin = max(tmin, t0);
    tmax = min(tmax, t1);
    if (tmin > tmax) {
        return false;
    }

    t0 = (std::get<1>(minPoint) - std::get<1>(origin)) / std::get<1>(direction);
    t1 = (std::get<1>(maxPoint) - std::get<1>(origin)) / std::get<1>(direction);
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    tmin = max(tmin, t0);
    tmax = min(tmax, t1);
    if (tmin > tmax) {
        return false;
    }

    t0 = (std::get<2>(minPoint) - std::get<2>(origin)) / std::get<2>(direction);
    t1 = (std::get<2>(maxPoint) - std::get<2>(origin)) / std::get<2>(direction);
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    tmin = max(tmin, t0);
    tmax = min(tmax, t1);
    if (tmin > tmax) {
        return false;
    }

    return true;
}

int get_block(int x, int y, int z, std::unordered_map<std::tuple<int, int, int>, std::vector<uint64_t>, TupleHash, TupleEqual>& map_chunks) {
    //std::cout << x << " " << y << " " << z << std::endl;
    std::tuple<int, int, int> key = std::make_tuple(x/16, y/16, z/16);
    auto it = map_chunks.find(key);
    if (it != map_chunks.end()) {
        const std::vector<uint64_t>& chunk = map_chunks.find(key)->second;
        //std::cout << chunk.size() << std::endl;
        int cx = (16 + x % 16) % 16;
        int cy = (16 + y % 16) % 16;
        int cz = (16 + z % 16) % 16;
        
        if (((chunk[16 * cz + cy] >> (4 * cx)) & (static_cast<uint64_t>(0b1111))) == 0) {
            return 0;
        }
        else {
            //std::cout << ((chunk[16 * cz + cy] >> (4 * cx)) & (static_cast<uint64_t>(0b1111))) << std::endl;
            return ((chunk[16 * cz + cy] >> (4 * cx)) & (static_cast<uint64_t>(0b1111)));
        }
    }
    else {
        return 0;
    }
}

std::tuple<int,int,int> block_breaking(std::unordered_map<std::tuple<int, int, int>, std::vector<uint64_t>, TupleHash, TupleEqual>& map_chunks, float x_rotation, float y_rotation, float px, float py, float pz) {
    float a;
    
    if (cos(x_rotation)>0) {
        a = 1;
    }
    else{
        a = -1;
    }
    float b = tan(y_rotation);
    float c = a*tan(x_rotation);
    
    float magnitude = std::sqrt(a * a + c * c);
    a /= magnitude;
    c /= magnitude;
    std::vector<std::tuple<int, int, int>> possible_blocks = { std::make_tuple(0,0,0) };
    float step_size = 0.001;
    for (int i = 0; i < 10000;i++) {
        float t = i * step_size;
        float x = px + t * a;
        float y = py + t * b;
        float z = pz + t * c;
        if (!(std::get<0>(possible_blocks[possible_blocks.size()-1]) == floor(x) && std::get<1>(possible_blocks[possible_blocks.size() - 1]) == floor(y) && std::get<2>(possible_blocks[possible_blocks.size() - 1]) == floor(z))) {
            possible_blocks.push_back(std::make_tuple(x, y, z));
        }
    }
    possible_blocks.erase(possible_blocks.begin());
    for (auto ty : possible_blocks) {
        int x;
        int y;
        int z;
        std::tie(x, y, z) = ty;
        std::tuple<int, int, int> key = std::make_tuple(x / 16, y / 16, z / 16);
        std::vector<uint64_t>& chunk = map_chunks.find(key)->second;
        int cx = (16 + x % 16) % 16;
        int cy = (16 + y % 16) % 16;
        int cz = (16 + z % 16) % 16;
        if (get_block(x, y, z, map_chunks)!=0) {
            for (int i = 0; i < 10; i++) {
            }
            std::tuple<int, int, int> key = std::make_tuple(x / 16, y / 16, z / 16);
            std::vector<uint64_t>& chunk = map_chunks.find(key)->second;
            int cx = (16 + x % 16) % 16;
            int cy = (16 + y % 16) % 16;
            int cz = (16 + z % 16) % 16;
            chunk[16 * cz + cy] &= ~(static_cast<uint64_t>(0b1111) << (4 * cx));
            return std::make_tuple(x, y, z);
        }
    }
    return std::make_tuple(0, 0, 0);
}

std::tuple<int, int, int> block_placing(std::unordered_map<std::tuple<int, int, int>, std::vector<uint64_t>, TupleHash, TupleEqual>& map_chunks, float x_rotation, float y_rotation, float px, float py, float pz) {
    float a;
    if (cos(x_rotation) > 0) {
        a = 1;
    }
    else {
        a = -1;
    }
    float b = tan(y_rotation);
    float c = a * tan(x_rotation);
    float magnitude = std::sqrt(a * a + c * c);
    a /= magnitude;
    c /= magnitude;
    std::vector<std::tuple<int, int, int>> possible_blocks = { std::make_tuple(0,0,0) };
    float step_size = 0.001;
    for (int i = 0; i < 10000; i++) {
        float t = i * step_size;
        float x = px + t * a;
        float y = py + t * b;
        float z = pz + t * c;
        //get_block(x, y, z, map_chunks) != 0 && 
        if (!(std::get<0>(possible_blocks[possible_blocks.size() - 1]) == floor(x) && std::get<1>(possible_blocks[possible_blocks.size() - 1]) == floor(y) && std::get<2>(possible_blocks[possible_blocks.size() - 1]) == floor(z))) {
            possible_blocks.push_back(std::make_tuple(x, y, z));
        }
    }
    possible_blocks.erase(possible_blocks.begin());

    for (int i = 0; i < possible_blocks.size()-1;++i) {
        std::tuple<int, int, int>ty = possible_blocks[i+1];
        int x;
        int y;
        int z;
        std::tie(x, y, z) = ty;
        std::tuple<int, int, int> key = std::make_tuple(x / 16, y / 16, z / 16);
        std::vector<uint64_t>& chunk = map_chunks.find(key)->second;
        if (get_block(x, y, z, map_chunks) != 0) {
            std::tie(x, y, z) = possible_blocks[i];
            std::tuple<int, int, int> key = std::make_tuple(x / 16, y / 16, z / 16);
            std::vector<uint64_t>& chunk = map_chunks.find(key)->second;
            int cx = (16 + x % 16) % 16;
            int cy = (16 + y % 16) % 16;
            int cz = (16 + z % 16) % 16;
            chunk[16 * cz + cy] |= (static_cast<uint64_t>(0b1111) << (4 * cx));
            return std::make_tuple(x, y, z);
        }
    }
    return std::make_tuple(0, 0, 0);
}



int main() {
    //std::cout << "vey very pre" << std::endl;
    int(*screen)[4] = new int[characters_per_row * number_of_columns][4];
    //std::cout << "vey pre" << std::endl;
    fill_screen(screen);
    //std::cout << "pre" << std::endl;
    std::unordered_map<std::tuple<int, int, int>, std::vector<uint64_t>, TupleHash, TupleEqual> map_chunks;
    std::unordered_map<std::tuple<int, int, int>, std::unordered_map<std::tuple<int, int>, int, TupleHash2, TupleEqual2>, TupleHash, TupleEqual> map_triangles;
    std::vector<std::vector<int>> faces = {
        // Front face
        {0, 1, 2,20},
        {0, 3, 2,20},

        // Left face
        {0, 4, 7,22},
        {0, 3, 7,22},

        // Bottom face
        {0, 1, 5,24},
        {0, 4, 5,24},

        // Back face
        {4, 5, 6,26},
        {4, 7, 6,26},

        // Right face
        {1, 5, 6,28},
        {1, 2, 6,28},

        // Top face
        {3, 2, 6,30},
        {3, 7, 6,30}
    };
    std::vector<std::vector<int>> vertices = {
        {0,0,0},
        {1,0,0},
        {1,1,0},
        {0,1,0},
        {0,0,1},
        {1,0,1},
        {1,1,1},
        {0,1,1},
    };
    std::vector<std::vector<int>> UV_vertices = { {0,0},{1,0},{0,1},{1,1} };
    srand(static_cast<unsigned int>(time(nullptr)));
    auto last_time = std::chrono::steady_clock::now();
    int render_distance = 8;
    //std::cout << "start" << std::endl;
    while (true) {
        //std::cout << "start" << std::endl;

        for (int x = -render_distance; x <= render_distance; x++) {
            for (int y = -render_distance; y <= render_distance; y++) {
                for (int z = -render_distance; z <= render_distance; z++) {
                    //std::cout << x << " " << y << " " << z << std::endl;
                    std::tuple<int, int, int> key = std::make_tuple(px / 16 + x, py / 16 + y, pz / 16 + z);
                    std::unordered_map<std::tuple<int, int, int>, std::vector<uint64_t>, TupleHash, TupleEqual>::iterator it = map_chunks.find(key);
                    if (it == map_chunks.end()) {
                        map_chunks[key] = make_chunk(px / 16 + x, py / 16 + y, pz / 16 + z);
                        std::vector<uint64_t> chunk = map_chunks[key];
                        if (x < render_distance || y < render_distance || z < render_distance) {
                            //map_triangles[key] = chunk_to_triangles(map_chunks, px / 16 + x, py / 16 + y, pz / 16 + z);
                        }
                    }
                }
            }
        }
        //std::cout << "start" << std::endl;

        for (int x = 1 - render_distance; x <= render_distance - 1; x++) {
            for (int y = 1-render_distance; y <= render_distance-1; y++) {
                for (int z = 1-render_distance; z <= render_distance-1; z++) {
                    //std::cout << x << " " << y << " " << z << std::endl;
                    std::tuple<int, int, int> key = std::make_tuple(px / 16 + x, py / 16 + y, pz / 16 + z);
                    std::unordered_map<std::tuple<int, int, int>, std::unordered_map<std::tuple<int, int>, int, TupleHash2, TupleEqual2>, TupleHash, TupleEqual>::iterator it = map_triangles.find(key);
                    if (it == map_triangles.end()) {
                        //std::vector<uint64_t> chunk = map_chunks[key];
                        map_triangles[key] = chunk_to_triangles(map_chunks, px / 16 + x, py / 16 + y, pz / 16 + z);
                    }
                }
            }
        }
        //std::cout << "start" << std::endl;

        std::vector<std::tuple<int, int, int>> keysToRemove;
        for (auto& pair : map_chunks) {
            const std::tuple<int, int, int>& co = pair.first;
            if (abs(std::get<0>(co) - static_cast<int>(px) / 16) > render_distance || abs(std::get<1>(co) - static_cast<int>(py) / 16) > render_distance || abs(std::get<2>(co) - static_cast<int>(pz) / 16) > render_distance) {
                keysToRemove.push_back(co);
            }
        }
        for (const auto& key : keysToRemove) {
            map_chunks.erase(key);
            map_triangles.erase(key);
        }
        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<float> delta_seconds = current_time - last_time;
        last_time = current_time;
        float delta_time = delta_seconds.count();
        std::cout << delta_time << '\n';

        //BLOCK BREAKING
        if (GetAsyncKeyState(VK_LBUTTON) & 0x8000) {
            std::tuple<int, int, int> key2 = block_breaking(map_chunks, M_PI / 2 - x_rotation, y_rotation, px, py, pz);
            int x2, y2, z2;
            std::tie(x2, y2, z2) = key2;
            std::vector<std::vector<int>> blocks = { {x2,y2,z2},{x2 + 1,y2 + 0,z2 + 0},{x2 - 1,y2 + 0,z2 + 0},{x2 + 0,y2 + 1,z2 + 0},{x2 + 0,y2 - 1,z2 + 0},{x2 + 0,y2 + 0,z2 + 1},{x2 + 0,y2 + 0,z2 - 1} };
            for (std::vector<int> block : blocks) {
                auto& triangles = map_triangles[std::make_tuple(block[0] / 16, block[1] / 16, block[2] / 16)];
                block_to_triangles(triangles, block, map_chunks, faces, UV_vertices, vertices);
            }
        }

        //BLOCK PLACING
        if (GetAsyncKeyState(VK_RBUTTON) & 0x8000) {
            std::tuple<int, int, int> key2 = block_placing(map_chunks, M_PI / 2 - x_rotation, y_rotation, px, py, pz);
            int x2, y2, z2;
            std::tie(x2, y2, z2) = key2;
            std::vector<std::vector<int>> blocks = { {x2,y2,z2},{x2 + 1,y2 + 0,z2 + 0},{x2 - 1,y2 + 0,z2 + 0},{x2 + 0,y2 + 1,z2 + 0},{x2 + 0,y2 - 1,z2 + 0},{x2 + 0,y2 + 0,z2 + 1},{x2 + 0,y2 + 0,z2 - 1} };
            for (std::vector<int> block : blocks) {
                auto& triangles = map_triangles[std::make_tuple(block[0] / 16, block[1] / 16, block[2] / 16)];
                block_to_triangles(triangles, block, map_chunks, faces, UV_vertices, vertices);
            }
        }
        //std::cout << "controls" << std::endl;
        controls(x_rotation, y_rotation, px, py, pz, min(0.1, delta_time), blocks_from_neighboring_chunks(map_chunks, px, py, pz));
        //std::cout << "update" << std::endl;
        update_screen(screen, map_triangles, x_rotation, y_rotation, px, py, pz);
        //std::cout << "draw" << std::endl;
        draw_screen(screen);
    }

    return 0;
}

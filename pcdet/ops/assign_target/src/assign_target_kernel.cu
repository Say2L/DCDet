#include <math.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
// #define DEBUG


__device__ inline void lidar_to_local_coords(float shift_x, float shift_y, float rot_angle, float &local_x, float &local_y){
    float cosa = cos(-rot_angle), sina = sin(-rot_angle);
    local_x = shift_x * cosa + shift_y * (-sina);
    local_y = shift_x * sina + shift_y * cosa;
}


__device__ inline int check_pt_in_box2d(const float *pt, const float *box2d, float &local_x, float &local_y){
    // param pt: (x, y)
    // param box2d: [x, y, dx, dy, heading] (x, y) is the box center

    const float MARGIN = 1e-5;
    float x = pt[0], y = pt[1];
    float cx = box2d[0], cy = box2d[1];
    float dx = box2d[2], dy = box2d[3], rz = box2d[4];

    lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);
    float in_flag = (fabs(local_x) < dx / 2.0 + MARGIN) & (fabs(local_y) < dy / 2.0 + MARGIN);
    return in_flag;
}

__device__ inline bool check_pt_in_cross_area(int pointx, int pointy, const int *centerx, const int *centery, int r){
    // param pt: (x, y)
    // param box2d: [x, y, dx, dy, heading] (x, y) is the box center
    bool in_flag = ((fabs(pointx - centerx[0]) + fabs(pointy - centery[0])) <= r);
    return in_flag;
}


__device__ inline int check_pt_in_box2d_center(const float *pt, const float *box2d, float &local_x, float &local_y){
    // param pt: (x, y)
    // param box2d: [x, y, dx, dy, heading] (x, y) is the box center

    const float MARGIN = 1e-5;
    float x = pt[0], y = pt[1];
    float cx = box2d[0], cy = box2d[1];

    float in_flag = (fabs(cx - x) <= 2.0 + MARGIN) & (fabs(cy - y) <= 2.0 + MARGIN);
    local_x = x - cx;
    local_y = y - cy;
    return in_flag;
}


__device__ inline int check_positive_in_box2d(const float *pt, const float *box2d, float &local_x, float &local_y){
    // param pt: (x, y)
    // param box2d: [x, y, dx, dy, heading] (x, y) is the box center

    const float MARGIN = 1e-5;
    float x = pt[0], y = pt[1];
    float cx = box2d[0], cy = box2d[1];
    float dx = box2d[2], dy = box2d[3], rz = box2d[4];

    lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);
    float in_flag = ((pow(local_x, 2.0) / pow(dx / 2, 2.0) + pow(local_y, 2.0) / pow(dy / 2, 2.0))) <= 1;
    return in_flag;
}

__global__ void points_in_boxes_kernel(int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points){
    // params boxes: (N, 7) [x, y, dx, dy, heading] (x, y) is the box center
    // params pts: (npoints, 2) [x, y] in grid_map
    // params boxes_idx_of_points: (N, npoints), default 0

    int box_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (box_idx >= boxes_num || pt_idx >= pts_num) return;

    boxes += box_idx * 5;
    pts += pt_idx * 2;
    box_idx_of_points += box_idx * pts_num + pt_idx;

    float local_x = 0, local_y = 0;
    int cur_in_flag = 0;
    
    cur_in_flag = check_pt_in_box2d(pts, boxes, local_x, local_y);
    if (cur_in_flag){
        box_idx_of_points[0] = 1;
    }
}

__global__ void points_in_cross_area_kernel(int boxes_num, int sizex, int sizey, const int r, const int *centerx,
    const int *centery, int *box_idx_of_points){
    // params boxes: (N, 7) [x, y, dx, dy, heading] (x, y) is the box center
    // params pts: (npoints, 2) [x, y] in grid_map
    // params boxes_idx_of_points: (N, npoints), default 0

    int box_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (box_idx >= boxes_num || pt_idx >= sizex * sizey ) return;

    centerx += box_idx;
    centery += box_idx;
    int pointy = pt_idx / sizex;
    int pointx = pt_idx % sizex;
    box_idx_of_points += box_idx * sizex * sizey + pt_idx;

    bool cur_in_flag = check_pt_in_cross_area(pointx, pointy, centerx, centery, r);
    if (cur_in_flag){
        box_idx_of_points[0] = 1;
    }
}

__global__ void positive_points_in_boxes_kernel(int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points){
    // params boxes: (N, 7) [x, y, dx, dy, heading] (x, y) is the box center
    // params pts: (npoints, 2) [x, y] in grid_map
    // params boxes_idx_of_points: (N, npoints), default 0

    int box_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (box_idx >= boxes_num || pt_idx >= pts_num) return;

    boxes += box_idx * 5;
    pts += pt_idx * 2;
    box_idx_of_points += box_idx * pts_num + pt_idx;

    float local_x = 0, local_y = 0;
    int cur_in_flag = 0;
    
    cur_in_flag = check_positive_in_box2d(pts, boxes, local_x, local_y);
    if (cur_in_flag){
        box_idx_of_points[0] = 1;
    }
}

__global__ void heatmap_in_boxes_kernel(int boxes_num, int pts_num, const float *boxes,
    const float *pts, float *box_idx_of_points){
    // params boxes: (N, 5) [x, y, dx, dy, heading] (x, y) is the box center
    // params pts: (npoints, 2) [x, y] in grid_map
    // params boxes_idx_of_points: (N, npoints), default 0

    int box_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (box_idx >= boxes_num || pt_idx >= pts_num) return;

    boxes += box_idx * 5;
    pts += pt_idx * 2;
    box_idx_of_points += box_idx * pts_num + pt_idx;
    
    float dim_x = boxes[2], dim_y = boxes[3];
    float local_x = 0, local_y = 0;
    int cur_in_flag = 0;
    
    cur_in_flag = check_pt_in_box2d(pts, boxes, local_x, local_y);
    if (cur_in_flag){
        box_idx_of_points[0] = exp( - (local_x * local_x + local_y * local_y * (dim_x / dim_y) * (dim_x / dim_y)) / (sqrt(dim_x * dim_x + dim_y * dim_y)));
    }
}

__global__ void heatmap_in_boxes_center_kernel(int boxes_num, int pts_num, const float *boxes,
    const float *pts, float *box_idx_of_points){
    // params boxes: (N, 5) [x, y, dx, dy, heading] (x, y) is the box center
    // params pts: (npoints, 2) [x, y] in grid_map
    // params boxes_idx_of_points: (N, npoints), default 0

    int box_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (box_idx >= boxes_num || pt_idx >= pts_num) return;

    boxes += box_idx * 5;
    pts += pt_idx * 2;
    box_idx_of_points += box_idx * pts_num + pt_idx;
    
    float local_x = 0, local_y = 0;
    int cur_in_flag = 0;
    
    cur_in_flag = check_pt_in_box2d_center(pts, boxes, local_x, local_y);
    if (cur_in_flag){
        box_idx_of_points[0] = exp( - (local_x * local_x + local_y * local_y)  / 2);
    }
}

void points_in_boxes_launcher(int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points){
     // params boxes: (N, 7) [x, y, dx, dy, heading] (x, y) is the box center
    // params pts: (npoints, 2) [x, y] in grid_map
    // params boxes_idx_of_points: (N, npoints), default 0
    cudaError_t err;

    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num);
    dim3 threads(THREADS_PER_BLOCK);
    points_in_boxes_kernel<<<blocks, threads>>>(boxes_num, pts_num, boxes, pts, box_idx_of_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}

void points_in_cross_area_launcher(int boxes_num, int sizex, int sizey, int r, const int *centerx,
     const int *centery, int *box_idx_of_points){
     // params boxes: (N, 7) [x, y, dx, dy, heading] (x, y) is the box center
    // params pts: (npoints, 2) [x, y] in grid_map
    // params boxes_idx_of_points: (N, npoints), default 0
    cudaError_t err;

    dim3 blocks(DIVUP(sizex * sizey, THREADS_PER_BLOCK), boxes_num);
    dim3 threads(THREADS_PER_BLOCK);
    

    points_in_cross_area_kernel<<<blocks, threads>>>(boxes_num, sizex, sizey, r, centerx, centery, box_idx_of_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}

void positive_points_in_boxes_launcher(int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points){
     // params boxes: (N, 7) [x, y, dx, dy, heading] (x, y) is the box center
    // params pts: (npoints, 2) [x, y] in grid_map
    // params boxes_idx_of_points: (N, npoints), default 0
    cudaError_t err;

    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num);
    dim3 threads(THREADS_PER_BLOCK);
    positive_points_in_boxes_kernel<<<blocks, threads>>>(boxes_num, pts_num, boxes, pts, box_idx_of_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}

void heatmap_in_boxes_launcher(int boxes_num, int pts_num, const float *boxes,
    const float *pts, float *box_idx_of_points){
     // params boxes: (N, 7) [x, y, dx, dy, heading] (x, y) is the box center
    // params pts: (npoints, 2) [x, y] in grid_map
    // params boxes_idx_of_points: (N, npoints), default 0
    cudaError_t err;

    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num);
    dim3 threads(THREADS_PER_BLOCK);
    heatmap_in_boxes_kernel<<<blocks, threads>>>(boxes_num, pts_num, boxes, pts, box_idx_of_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}

void heatmap_in_boxes_center_launcher(int boxes_num, int pts_num, const float *boxes,
    const float *pts, float *box_idx_of_points){
     // params boxes: (N, 7) [x, y, dx, dy, heading] (x, y) is the box center
    // params pts: (npoints, 2) [x, y] in grid_map
    // params boxes_idx_of_points: (N, npoints), default 0
    cudaError_t err;

    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num);
    dim3 threads(THREADS_PER_BLOCK);
    heatmap_in_boxes_center_kernel<<<blocks, threads>>>(boxes_num, pts_num, boxes, pts, box_idx_of_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}

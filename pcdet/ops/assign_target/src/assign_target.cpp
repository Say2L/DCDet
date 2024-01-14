#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <assert.h>

void points_in_boxes_launcher(int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points);

void points_in_cross_area_launcher(int boxes_num, int sizex, int sizey, int r, const int *centerx,
     const int *centery, int *box_idx_of_points);

void positive_points_in_boxes_launcher(int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points);

void heatmap_in_boxes_launcher(int boxes_num, int pts_num, const float *boxes,
    const float *pts, float *box_idx_of_points);

void heatmap_in_boxes_center_launcher(int boxes_num, int pts_num, const float *boxes,
    const float *pts, float *box_idx_of_points);

inline void lidar_to_local_coords_cpu(float shift_x, float shift_y, float rot_angle, float &local_x, float &local_y){
    float cosa = cos(-rot_angle), sina = sin(-rot_angle);
    local_x = shift_x * cosa + shift_y * (-sina);
    local_y = shift_x * sina + shift_y * cosa;
}


inline int check_pt_in_box2d_cpu(const float *pt, const float *box2d, float &local_x, float &local_y){
    // param pt: (x, y)
    // param box3d: [x, y, dx, dy, heading], (x, y) is the box center
    const float MARGIN = 1e-2;
    float x = pt[0], y = pt[1];
    float cx = box2d[0], cy = box2d[1];
    float dx = box2d[2], dy = box2d[3], rz = box2d[4];

    lidar_to_local_coords_cpu(x - cx, y - cy, rz, local_x, local_y);
    float in_flag = (fabs(local_x) < dx / 2.0 + MARGIN) & (fabs(local_y) < dy / 2.0 + MARGIN);
    return in_flag;
}


int points_in_boxes_cpu(at::Tensor boxes_tensor, at::Tensor pts_tensor, at::Tensor pts_indices_tensor){
    // params boxes: (N, 5) [x, y, dx, dy, heading], (x, y) is the box center, each box DO NOT overlaps
    // params pts: (num_points, 2) [x, y]
    // params pts_indices: (N, num_points)

//    CHECK_CONTIGUOUS(boxes_tensor);
//    CHECK_CONTIGUOUS(pts_tensor);
//    CHECK_CONTIGUOUS(pts_indices_tensor);

    int boxes_num = boxes_tensor.size(0);
    int pts_num = pts_tensor.size(0);

    const float *boxes = boxes_tensor.data<float>();
    const float *pts = pts_tensor.data<float>();
    int *pts_indices = pts_indices_tensor.data<int>();
    
    float local_x = 0, local_y = 0;
    for (int i = 0; i < boxes_num; i++){
        for (int j = 0; j < pts_num; j++){
            int cur_in_flag = check_pt_in_box2d_cpu(pts + j * 2, boxes + i * 5, local_x, local_y);
            pts_indices[i * pts_num + j] = cur_in_flag;
        }
    }

    return 1;
}


int points_in_boxes_gpu(at::Tensor boxes_tensor, at::Tensor pts_tensor, at::Tensor box_idx_of_points_tensor){
    // params boxes: (N, 7) [x, y, dx, dy, heading] (x, y) is the box center
    // params pts: (npoints, 2) [x, y]
    // params boxes_idx_of_points: (N, npoints), default -1

//    CHECK_INPUT(boxes_tensor);
//    CHECK_INPUT(pts_tensor);
//    CHECK_INPUT(box_idx_of_points_tensor);

    int boxes_num = boxes_tensor.size(0);
    int pts_num = pts_tensor.size(0);

    const float *boxes = boxes_tensor.data<float>();
    const float *pts = pts_tensor.data<float>();
    int *box_idx_of_points = box_idx_of_points_tensor.data<int>();

    points_in_boxes_launcher(boxes_num, pts_num, boxes, pts, box_idx_of_points);

    return 1;
}

int points_in_cross_area_gpu(at::Tensor centerx_tensor, at::Tensor centery_tensor, at::Tensor box_idx_of_points_tensor, int r){
    // params boxes: (N, 7) [x, y, dx, dy, heading] (x, y) is the box center
    // params pts: (npoints, 2) [x, y]
    // params boxes_idx_of_points: (N, npoints), default -1

//    CHECK_INPUT(boxes_tensor);
//    CHECK_INPUT(pts_tensor);
//    CHECK_INPUT(box_idx_of_points_tensor);

    int boxes_num = box_idx_of_points_tensor.size(0);
    int sizey =  box_idx_of_points_tensor.size(1);
    int sizex = box_idx_of_points_tensor.size(2);

    const int *centerx = centerx_tensor.data<int>();
    const int *centery = centery_tensor.data<int>();
    int *box_idx_of_points = box_idx_of_points_tensor.data<int>();

    points_in_cross_area_launcher(boxes_num, sizex, sizey, r, centerx, centery, box_idx_of_points);

    return 1;
}

int positive_points_in_boxes_gpu(at::Tensor boxes_tensor, at::Tensor pts_tensor, at::Tensor box_idx_of_points_tensor){
    // params boxes: (N, 7) [x, y, dx, dy, heading] (x, y) is the box center
    // params pts: (npoints, 2) [x, y]
    // params boxes_idx_of_points: (N, npoints), default -1

//    CHECK_INPUT(boxes_tensor);
//    CHECK_INPUT(pts_tensor);
//    CHECK_INPUT(box_idx_of_points_tensor);

    int boxes_num = boxes_tensor.size(0);
    int pts_num = pts_tensor.size(0);

    const float *boxes = boxes_tensor.data<float>();
    const float *pts = pts_tensor.data<float>();
    int *box_idx_of_points = box_idx_of_points_tensor.data<int>();

    positive_points_in_boxes_launcher(boxes_num, pts_num, boxes, pts, box_idx_of_points);

    return 1;
}

int heatmap_in_boxes_gpu(at::Tensor boxes_tensor, at::Tensor pts_tensor, at::Tensor box_idx_of_points_tensor){
    // params boxes: (N, 7) [x, y, dx, dy, heading] (x, y) is the box center
    // params pts: (npoints, 2) [x, y]
    // params boxes_idx_of_points: (N, npoints), default -1

//    CHECK_INPUT(boxes_tensor);
//    CHECK_INPUT(pts_tensor);
//    CHECK_INPUT(box_idx_of_points_tensor);

    int boxes_num = boxes_tensor.size(0);
    int pts_num = pts_tensor.size(0);

    const float *boxes = boxes_tensor.data<float>();
    const float *pts = pts_tensor.data<float>();
    float *box_idx_of_points = box_idx_of_points_tensor.data<float>();

    heatmap_in_boxes_launcher(boxes_num, pts_num, boxes, pts, box_idx_of_points);

    return 1;
}

int heatmap_in_boxes_center_gpu(at::Tensor boxes_tensor, at::Tensor pts_tensor, at::Tensor box_idx_of_points_tensor){
    // params boxes: (N, 7) [x, y, dx, dy, heading] (x, y) is the box center
    // params pts: (npoints, 2) [x, y]
    // params boxes_idx_of_points: (N, npoints), default -1

//    CHECK_INPUT(boxes_tensor);
//    CHECK_INPUT(pts_tensor);
//    CHECK_INPUT(box_idx_of_points_tensor);

    int boxes_num = boxes_tensor.size(0);
    int pts_num = pts_tensor.size(0);

    const float *boxes = boxes_tensor.data<float>();
    const float *pts = pts_tensor.data<float>();
    float *box_idx_of_points = box_idx_of_points_tensor.data<float>();

    heatmap_in_boxes_center_launcher(boxes_num, pts_num, boxes, pts, box_idx_of_points);

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("points_in_boxes_cpu", &points_in_boxes_cpu, "points_in_boxes_cpu");
    m.def("points_in_boxes_gpu", &points_in_boxes_gpu, "points_in_boxes_gpu");
    m.def("positive_points_in_boxes_gpu", &positive_points_in_boxes_gpu, "positive_points_in_boxes_gpu");
    m.def("heatmap_in_boxes_gpu", &heatmap_in_boxes_gpu, "heatmap_in_boxes_gpu");
    m.def("heatmap_in_boxes_center_gpu", &heatmap_in_boxes_center_gpu, "heatmap_in_boxes_center_gpu");
    m.def("points_in_cross_area_gpu", &points_in_cross_area_gpu, "points_in_cross_area_gpu");
}
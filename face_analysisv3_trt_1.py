import cv2
import numpy as np
import os
from process import resize_image, postprocess, draw_faces, reproject_points
import fast_face_align as face_align
import sklearn.preprocessing as sk
from trt_loader import TrtModel
import cupy as cp

def _normalize_on_device(g_img, mean=0., std=1.):
    g_img = np.transpose(g_img, (0, 3, 1, 2))
    g_img = np.subtract(g_img, mean, dtype=np.float32)
    g_img = np.multiply(g_img, 1 / std)
    return g_img


def load_trt(path):
    model = TrtModel(path)
    model.build()
    return model

def net_detect(data, model):
    #run model 
    response = model.run(input=data, deflatten=True, as_dict=True, from_device=False, infer_shape=data.shape)
    #get response from output
    heatmap = response['537']
    scale = response['538']
    offset = response['539']
    lms = response['540']
    return heatmap, scale, offset, lms

def net_rec(data, model):
    #run model
    response = model.run(input=data, deflatten=True, as_dict=True, from_device=False, infer_shape=data.shape)
    #get response from output
    embeddings = response['fc1']
    return embeddings


def align_and_crop(heatmaps, scales, offsets, lms, scale_factors, org_imgs):
    
    # Postprocess and Alignment
    all_crops=[]
    for heatmap, scale, offset, lm, scale_factor, image in zip(heatmaps, scales, offsets, lms, scale_factors, org_imgs):
        bbox, landm = postprocess(np.expand_dims(heatmap, axis=0), np.expand_dims(lm, axis=0), 
                                  np.expand_dims(offset, axis=0), np.expand_dims(scale,axis=0), 
                                  (640, 480), threshold, nms_threshold)
        # Scale detected boxes and points to coordinates of original image
        bbox_ = reproject_points(bbox[:4], scale=scale_factor)
        landm_ = reproject_points(landm, scale=scale_factor)
        score = bbox[:, 4]
        
        pt1 = tuple(bbox_[0][0:2].astype(int))
        pt2 = tuple(bbox_[0][2:4].astype(int))
        crop = image[max(0, int(pt1[1])):max(0, int(pt2[1])), max(0, int(pt1[0])):max(0, int(pt2[0])), :]
        res_image, scale_f = resize_image(crop, (112, 112))
#         cv2.imwrite('dd.jpg', crop)
#         print(crop.shape)
        all_crops.append(res_image)

    return all_crops


if __name__ == '__main__':

    # Read and resize image - DO IT ONCE
    det_path = '/workspace/bharat/cdot/face_analysis_final/triton_model_repository/trt_fp16_centerface/1/model.plan'
    rec_path = '/workspace/bharat/cdot/face_analysis_final/triton_model_repository/trt_fp16_arcface/1/model.plan'
    
    det_model = load_trt(det_path)
    rec_model = load_trt(rec_path)
    
    
    
    detect_size = [640, 480]
    threshold = 0.4
    nms_threshold = 0.3 
    
    files = os.listdir('data2')
    all_imgs = []
    scale_factors = []
    org_imgs = []
    for file in files:
        path = os.path.join('data2', file)
        image = cv2.imread(path)
        res_image, scale_factor = resize_image(image, detect_size)
        reshape_img = res_image[:, :, (2, 1, 0)].transpose(2, 0, 1)
        org_imgs.append(image)
        all_imgs.append(reshape_img)
        scale_factors.append(scale_factor)

    #Batch Input
    input_batch = np.array(all_imgs).astype("float32")

    #centerface
    heatmaps, scales, offsets, lms = net_detect(input_batch, det_model)
    
    #align and crop
    all_crops = align_and_crop(heatmaps, scales, offsets, lms, scale_factors, org_imgs)
    
    face_img = np.stack(all_crops)
    print(face_img.shape)
    
    stream = None
    input_ptr = None
    
    face_img = _normalize_on_device(face_img, mean=0,
                                       std=1)
    #Arcface
    embeddings = net_rec(face_img, rec_model)
    
    #Normalize
    norm_embeddings = sk.normalize(embeddings, axis=1)
    
    #Save
    np.save("arcface_embeddings1.npy", norm_embeddings)
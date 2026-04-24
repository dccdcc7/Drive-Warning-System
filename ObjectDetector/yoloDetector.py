import os
import cv2
import random
import logging
import numpy as np
from typing import *
try :
	import sys
	from utils import ObjectModelType, hex_to_rgb, NMS, Scaler
	from core import ObjectDetectBase, RectInfo
	sys.path.append("..")
	from coreEngine import TensorRTEngine, OnnxEngine
except :
	from .utils import ObjectModelType, hex_to_rgb, NMS, Scaler
	from .core import ObjectDetectBase, RectInfo
	from coreEngine import TensorRTEngine, OnnxEngine

class YoloLiteParameters():
	def __init__(self, model_type, input_shape, num_classes):
		self.lite = False
		if (model_type == ObjectModelType.YOLOV5_LITE) :
			self.lite = True
		anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
		self.nl = len(anchors)
		self.na = len(anchors[0]) // 2
		self.no = num_classes + 5
		self.grid = [np.zeros(1)] * self.nl
		self.stride = np.array([8., 16., 32.])
		self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
		self.input_shape = input_shape[-2:]

	def __make_grid(self, nx=20, ny=20):
		xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
		return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

	def lite_postprocess(self, outs):
		if self.lite :
			row_ind = 0
			for i in range(self.nl):
				h, w = int(self.input_shape[0] / self.stride[i]), int(self.input_shape[1] / self.stride[i])
				length = int(self.na * h * w)
				if self.grid[i].shape[2:4] != (h, w):
					self.grid[i] = self.__make_grid(w, h)

				outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
					self.grid[i], (self.na, 1))) * int(self.stride[i])
				outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
					self.anchor_grid[i], h * w, axis=0)
				row_ind += length
		return outs

# class YoloDetector(ObjectDetectBase, YoloLiteParameters):
# 	_defaults = {
# 		"model_path": './models/yolov5n-coco.onnx',
# 		"model_type" : ObjectModelType.YOLOV5,
# 		"classes_path" : './models/coco_label.txt',
# 		"box_score" : 0.4,
# 		"box_nms_iou" : 0.45
# 	}

# 	def __init__(self, logger=None, **kwargs):
# 		ObjectDetectBase.__init__(self, logger)
# 		self.__dict__.update(kwargs) # and update with user overrides

# 		self._initialize_class(self.classes_path)
# 		self._initialize_model(self.model_path)
# 		YoloLiteParameters.__init__(self, self.model_type, self.input_shapes, len(self.class_names))

# 	def _initialize_model(self, model_path : str) -> None:
# 		model_path = os.path.expanduser(model_path)
# 		if (self.logger) :
# 			self.logger.debug("model path: %s." % model_path)

# 		if model_path.endswith('.trt') :
# 			self.engine = TensorRTEngine(model_path)
# 		else :
# 			self.engine = OnnxEngine(model_path)

# 		if (self.logger) :
# 			self.logger.info(f'YoloDetector Type : [{self.engine.framework_type}] || Version : [{self.engine.providers}]')
# 		self.set_input_details(self.engine)
# 		self.set_output_details(self.engine)

# 	def _initialize_class(self, classes_path : str) -> None:
# 		classes_path = os.path.expanduser(self.classes_path)
# 		if (self.logger) :
# 			self.logger.debug("class path: %s." % classes_path)
# 		assert os.path.isfile(classes_path), Exception("%s is not exist." % classes_path)

# 		with open(classes_path) as f:
# 			class_names = f.readlines()
# 		self.class_names = [c.strip() for c in class_names]
# 		get_colors = list(map(lambda i: hex_to_rgb("#" +"%06x" % random.randint(0, 0xFFFFFF)), range(len(self.class_names)) ))
# 		self.colors_dict = dict(zip(list(self.class_names), get_colors))

# 	def __prepare_input(self, srcimg : cv2) -> Tuple[np.ndarray, Scaler] :
# 		scaler = Scaler(self.input_shapes[-2:], True)
# 		image = scaler.process_image(srcimg)
# 		# HWC -> NCHW format
# 		blob = cv2.dnn.blobFromImage(image, 1/255.0, (image.shape[1], image.shape[0]), 
# 										swapRB=True, crop=False).astype(self.input_types)
# 		return blob, scaler

# 	def __process_output(self, output: np.ndarray) -> Tuple[List[np.ndarray,], list, list, list]:
# 		_raw_boxes = []
# 		_raw_kpss = []
# 		_raw_class_ids = []
# 		_raw_class_confs = []

# 		'''
# 		YOLOv5/6/7 outputs shape -> (-1, obj_conf + 5[bbox, cls_conf])
# 		YOLOv8/9 outputs shape -> (obj_conf + 4[bbox], -1)
# 		'''
# 		if (self.model_type in [ObjectModelType.YOLOV8, ObjectModelType.YOLOV9, ObjectModelType.YOLOV10]) :
# 			output = output.T
		
# 		output = self.lite_postprocess(output)

# 		# inference output
# 		for detection in output:
# 			if (self.model_type in [ObjectModelType.YOLOV8, ObjectModelType.YOLOV9, ObjectModelType.YOLOV10]) :
# 				obj_cls_probs = detection[4:]
# 			else :
# 				obj_cls_probs = detection[5:] * detection[4] # cls_conf * obj_conf 

# 			classId = np.argmax(obj_cls_probs)
# 			classConf = float(obj_cls_probs[classId])
# 			if classConf > self.box_score :
# 				x, y, w, h = detection[0:4]
# 				_raw_class_ids.append(classId)
# 				_raw_class_confs.append(classConf)
# 				_raw_boxes.append(np.stack([(x - 0.5 * w), (y - 0.5 * h), (x + 0.5 * w), (y + 0.5 * h)], axis=-1))
# 		return _raw_boxes, _raw_class_ids, _raw_class_confs, _raw_kpss

# 	def get_nms_results(self, boxes : np.array, class_confs : list, class_ids : list, kpss : np.array) -> List[Tuple[list, list]]:
# 		results = []
# 		# nms_results = cv2.dnn.NMSBoxes(boxes, class_confs, self.box_score, self.box_nms_iou) 
# 		# nms_results = NMS.fast_nms(boxes, class_confs, self.box_nms_iou, "xywh") 
# 		nms_results = NMS.fast_soft_nms(boxes, class_confs, self.box_nms_iou, dets_type="xywh") 

# 		if len(nms_results) > 0:
# 			for i in nms_results:
# 				try :
# 					predicted_class = self.class_names[class_ids[i]] 
# 				except :
# 					predicted_class = "unknown"
# 				conf = class_confs[i]
# 				bbox = boxes[i]

# 				kpsslist = []
# 				if (kpss.size != 0) :
# 					for j in range(5):
# 						kpsslist.append( ( int(kpss[i, j, 0]) , int(kpss[i, j, 1]) ) )
# 				results.append(RectInfo(*bbox, conf=conf, 
# 											   label=predicted_class,
# 											   kpss=kpsslist))
# 		return results

# 	def DetectFrame(self, srcimg : cv2) -> None:
# 		input_tensor, scaler = self.__prepare_input(srcimg)

# 		output_from_network = self.engine.engine_inference(input_tensor)[0].squeeze(axis=0)

# 		_raw_boxes, _raw_class_ids, _raw_class_confs, _raw_kpss = self.__process_output(output_from_network)
		
# 		transform_boxes = scaler.convert_boxes_coordinate(_raw_boxes)
# 		transform_kpss = scaler.convert_kpss_coordinate(_raw_kpss)
# 		self._object_info = self.get_nms_results(transform_boxes, _raw_class_confs, _raw_class_ids, transform_kpss)

# 	def DrawDetectedOnFrame(self, frame_show : cv2) -> None:
# 		tl = 3 or round(0.002 * (frame_show.shape[0] + frame_show.shape[1]) / 2) + 1    # line/font thickness
# 		if ( len(self._object_info) != 0 )  :
# 			for _info in self._object_info:
# 				xmin, ymin, xmax, ymax = _info.tolist()
# 				label = _info.label
				
# 				if (len(_info.kpss) != 0) :
# 					for kp in _info.kpss :
# 						cv2.circle(frame_show,  kp, 1, (255, 255, 255), thickness=-1)
# 				c1, c2 = (xmin, ymin), (xmax, ymax)        
# 				tf = max(tl - 1, 1)  # font thickness
# 				t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
# 				c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

# 				if (label != 'unknown') :
# 					cv2.rectangle(frame_show, c1, c2, self.colors_dict[label], -1, cv2.LINE_AA)
# 					self.cornerRect(frame_show, _info.tolist(), colorR=self.colors_dict[label], colorC=self.colors_dict[label])
# 				else :
# 					cv2.rectangle(frame_show, c1, c2, (0, 0, 0), -1, cv2.LINE_AA)
# 					self.cornerRect(frame_show, _info.tolist(), colorR= (0, 0, 0), colorC= (0, 0, 0) )
# 				cv2.putText(frame_show, label, (xmin + 2, ymin - 7), cv2.FONT_HERSHEY_TRIPLEX, tl / 4, (255, 255, 255), 2)


# 假设以下基类和枚举已经存在，根据你的上下文导入
# from your_module import ObjectDetectBase, YoloLiteParameters, ObjectModelType, RectInfo, NMS, Scaler, hex_to_rgb

class YoloDetector(ObjectDetectBase, YoloLiteParameters):
    _defaults = {
        "model_path": './models/yolo26n.onnx', # 默认路径改为v26
        "model_type": ObjectModelType.YOLO26,  # 默认类型设为v26
        "classes_path": './models/coco_label.txt',
        "box_score": 0.25,                      # v26通常置信度较低，调整默认值
        "box_nms_iou": 0.5
    }

    def __init__(self, logger=None, **kwargs):
        ObjectDetectBase.__init__(self, logger)
        self.__dict__.update(kwargs)  # update with user overrides

        self._initialize_class(self.classes_path)
        self._initialize_model(self.model_path)
        # 假设 YoloLiteParameters 需要知道输入形状和类别数量
        YoloLiteParameters.__init__(self, self.model_type, self.input_shapes, len(self.class_names))

    def _initialize_model(self, model_path: str) -> None:
        model_path = os.path.expanduser(model_path)
        if self.logger:
            self.logger.debug("model path: %s." % model_path)

        if model_path.endswith('.trt'):
            self.engine = TensorRTEngine(model_path)
        else:
            # YOLOv26 通常使用 CUDAExecutionProvider
            #self.engine = OnnxEngine(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.engine = OnnxEngine(model_path)

        if self.logger:
            self.logger.info(f'YoloDetector Type : [{self.engine.framework_type}] || Version : [{self.engine.providers}]')
        
        self.set_input_details(self.engine)
        self.set_output_details(self.engine)

    def _initialize_class(self, classes_path: str) -> None:
        classes_path = os.path.expanduser(self.classes_path)
        if self.logger:
            self.logger.debug("class path: %s." % classes_path)
        assert os.path.isfile(classes_path), Exception("%s is not exist." % classes_path)

        with open(classes_path) as f:
            class_names = f.readlines()
        self.class_names = [c.strip() for c in class_names]
        get_colors = list(map(lambda i: hex_to_rgb("#" + "%06x" % random.randint(0, 0xFFFFFF)), range(len(self.class_names))))
        self.colors_dict = dict(zip(list(self.class_names), get_colors))

    def __prepare_input(self, srcimg: cv2) -> Tuple[np.ndarray, Scaler]:
        # 使用 Scaler 进行预处理（保持比例，填充灰色）
        scaler = Scaler(self.input_shapes[-2:], True)
        image = scaler.process_image(srcimg)
        # HWC -> NCHW format
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (image.shape[1], image.shape[0]),
                                     swapRB=True, crop=False).astype(self.input_types)
        return blob, scaler

    def __process_output(self, output: np.ndarray) -> Tuple[List[np.ndarray], list, list, list]:
        _raw_boxes = []
        _raw_kpss = []
        _raw_class_ids = []
        _raw_class_confs = []

        # 针对 YOLOv26 的特殊处理
        if self.model_type == ObjectModelType.YOLO26:
            # YOLOv26 输出格式: [x1, y1, x2, y2, score, class_id]
            # output shape 通常为 (1, 25200, 6) 或类似，squeeze 后为 (25200, 6)
            # 注意：这里不需要像 v8 那样 transpose，也不需要像 v5 那样计算 obj_conf * cls_conf
            pass 
        else:
            # 兼容其他版本的逻辑
            if self.model_type in [ObjectModelType.YOLOV8, ObjectModelType.YOLOV9, ObjectModelType.YOLOV10]:
                output = output.T
            output = self.lite_postprocess(output)

        for detection in output:
            if self.model_type == ObjectModelType.YOLO26:
                # v26 格式: [x1, y1, x2, y2, conf, cls_id]
                confidence = detection[4]
                classId = int(detection[5])
                
                if confidence > self.box_score:
                    # 直接获取 x1, y1, x2, y2
                    x1, y1, x2, y2 = detection[0:4]
                    
                    _raw_class_ids.append(classId)
                    _raw_class_confs.append(float(confidence))
                    # 保存为 x1, y1, x2, y2 格式
                    _raw_boxes.append(np.array([x1, y1, x2, y2]))
            else:
                # 原有的 v5/v8 逻辑
                if self.model_type in [ObjectModelType.YOLOV8, ObjectModelType.YOLOV9, ObjectModelType.YOLOV10]:
                    obj_cls_probs = detection[4:]
                else:
                    obj_cls_probs = detection[5:] * detection[4]

                classId = np.argmax(obj_cls_probs)
                classConf = float(obj_cls_probs[classId])
                if classConf > self.box_score:
                    x, y, w, h = detection[0:4]
                    _raw_class_ids.append(classId)
                    _raw_class_confs.append(classConf)
                    _raw_boxes.append(np.stack([(x - 0.5 * w), (y - 0.5 * h), (x + 0.5 * w), (y + 0.5 * h)], axis=-1))
                    
        return _raw_boxes, _raw_class_ids, _raw_class_confs, _raw_kpss

    def get_nms_results(self, boxes: np.array, class_confs: list, class_ids: list, kpss: np.array):
        results = []
        
        # 注意：boxes 在 __process_output 中如果是 v26，已经是 xyxy 格式
        # 如果是 v5/v8，也是 xyxy 格式
        # 原代码 NMS.fast_soft_nms 使用的是 "xywh"，这里需要确认 NMS 函数的实现
        # 如果 NMS 函数强制要求 xywh，则需要在传入前转换，或者修改 NMS 调用参数
        # 为了安全起见，如果 boxes 是 xyxy，我们需要转换或者确保 NMS 支持 xyxy
        
        # 假设 NMS.fast_soft_nms 可以处理 xyxy 或者我们传入的 boxes 已经是正确的格式
        # 如果 NMS 需要 xywh:
        # boxes_xywh = np.copy(boxes)
        # boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0] # w
        # boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1] # h
        # nms_results = NMS.fast_soft_nms(boxes_xywh, class_confs, self.box_nms_iou, dets_type="xywh")
        
        # 这里直接传入 boxes，假设 NMS 逻辑适配或内部处理
        # 如果 NMS.fast_soft_nms 确实只支持 xywh，请务必按上述注释转换
        nms_results = NMS.fast_soft_nms(boxes, class_confs, self.box_nms_iou, dets_type="xywh") 

        if len(nms_results) > 0:
            for i in nms_results:
                try:
                    predicted_class = self.class_names[class_ids[i]]
                except:
                    predicted_class = "unknown"
                conf = class_confs[i]
                bbox = boxes[i]

                kpsslist = []
                if kpss.size != 0:
                    for j in range(5):
                        kpsslist.append((int(kpss[i, j, 0]), int(kpss[i, j, 1])))
                results.append(RectInfo(*bbox, conf=conf,
                                        label=predicted_class,
                                        kpss=kpsslist))
        return results

    def DetectFrame(self, srcimg: cv2) -> None:
        input_tensor, scaler = self.__prepare_input(srcimg)

        # 推理
        output_from_network = self.engine.engine_inference(input_tensor)[0].squeeze(axis=0)

        # 后处理
        _raw_boxes, _raw_class_ids, _raw_class_confs, _raw_kpss = self.__process_output(output_from_network)
        
        # 坐标转换 (将模型输出坐标映射回原图坐标)
        transform_boxes = scaler.convert_boxes_coordinate(_raw_boxes)
        transform_kpss = scaler.convert_kpss_coordinate(_raw_kpss)
        
        # NMS 和结果生成
        self._object_info = self.get_nms_results(transform_boxes, _raw_class_confs, _raw_class_ids, transform_kpss)
    
 
    def DetectFrameleft(self, srcimg: cv2) -> None:
        temp = srcimg.copy()
        black_width = int(temp.shape[1] * 0.216)
        if black_width > 0:
            temp[:, :black_width] = 0

        input_tensor, scaler = self.__prepare_input(temp)

        # 推理
        output_from_network = self.engine.engine_inference(input_tensor)[0].squeeze(axis=0)

        # 后处理
        _raw_boxes, _raw_class_ids, _raw_class_confs, _raw_kpss = self.__process_output(output_from_network)

        # 坐标转换 (将模型输出坐标映射回原图坐标)
        transform_boxes = scaler.convert_boxes_coordinate(_raw_boxes)
        transform_kpss = scaler.convert_kpss_coordinate(_raw_kpss)

        # NMS 和结果生成
        self._object_info = self.get_nms_results(transform_boxes, _raw_class_confs, _raw_class_ids, transform_kpss)

        h, w = srcimg.shape[:2]
        danger = False
        vehicle_labels = {
            "car", "truck", "bus", "motorbike", "motorcycle", "bicycle", "van"
        }
        min_bottom_ratio = 0.65
        min_area_ratio = 0.015
        if len(self._object_info) != 0:
            for _info in self._object_info:
                if _info.label not in vehicle_labels:
                    continue
                xmin, ymin, xmax, ymax = _info.tolist(dtype=float)
                box_area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
                if box_area <= 0:
                    continue
                bottom_ratio = ymax / h
                area_ratio = box_area / (w * h)
                if bottom_ratio >= min_bottom_ratio and area_ratio >= min_area_ratio:
                    danger = True
                    break				

        if danger:
            cv2.putText(srcimg, "danger", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 0, 255), 5)
        else:
            cv2.putText(srcimg, "safe", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 0), 5)


    def DetectFrameright(self, srcimg: cv2) -> None:
        temp = srcimg.copy()
        black_width = int(temp.shape[1] / 3)
        if black_width > 0:
            temp[:, -black_width:] = 0

        input_tensor, scaler = self.__prepare_input(temp)

        # 推理
        output_from_network = self.engine.engine_inference(input_tensor)[0].squeeze(axis=0)

        # 后处理
        _raw_boxes, _raw_class_ids, _raw_class_confs, _raw_kpss = self.__process_output(output_from_network)
        
        # 坐标转换 (将模型输出坐标映射回原图坐标)
        transform_boxes = scaler.convert_boxes_coordinate(_raw_boxes)
        transform_kpss = scaler.convert_kpss_coordinate(_raw_kpss)
        
        # NMS 和结果生成
        self._object_info = self.get_nms_results(transform_boxes, _raw_class_confs, _raw_class_ids, transform_kpss)

        h, w = srcimg.shape[:2]
        danger = False
        vehicle_labels = {
            "car", "truck", "bus", "motorbike", "motorcycle", "bicycle", "van"
        }
        min_bottom_ratio = 0.7
        min_area_ratio = 0.02
        if len(self._object_info) != 0:
            for _info in self._object_info:
                if _info.label not in vehicle_labels:
                    continue
                xmin, ymin, xmax, ymax = _info.tolist(dtype=float)
                box_area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
                if box_area <= 0:
                    continue
                bottom_ratio = ymax / h
                area_ratio = box_area / (w * h)
                if bottom_ratio >= min_bottom_ratio and area_ratio >= min_area_ratio:
                    danger = True
                    break

        if danger:
            cv2.putText(srcimg, "danger", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


    def DrawDetectedOnFrame(self, frame_show: cv2) -> None:
        tl = 3 or round(0.002 * (frame_show.shape[0] + frame_show.shape[1]) / 2) + 1  # line/font thickness
        if len(self._object_info) != 0:
            for _info in self._object_info:
                xmin, ymin, xmax, ymax = _info.tolist()
                label = _info.label

                if len(_info.kpss) != 0:
                    for kp in _info.kpss:
                        cv2.circle(frame_show, kp, 1, (255, 255, 255), thickness=-1)
                c1, c2 = (xmin, ymin), (xmax, ymax)
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

                if label != 'unknown':
                    color = self.colors_dict.get(label, (0, 0, 255)) # 获取颜色，默认红色
                    cv2.rectangle(frame_show, c1, c2, color, -1, cv2.LINE_AA)
                    self.cornerRect(frame_show, _info.tolist(), colorR=color, colorC=color)
                else:
                    cv2.rectangle(frame_show, c1, c2, (0, 0, 0), -1, cv2.LINE_AA)
                    self.cornerRect(frame_show, _info.tolist(), colorR=(0, 0, 0), colorC=(0, 0, 0))
                cv2.putText(frame_show, label, (xmin + 2, ymin - 7), cv2.FONT_HERSHEY_TRIPLEX, tl / 4, (255, 255, 255), 2)


if __name__ == "__main__":
	import time
	import sys

	capture = cv2.VideoCapture(r"../test.mp4")
	config = {
		"model_path": './models/yolo26s.onnx',
		"model_type" : ObjectModelType.YOLO26,
		"classes_path" : 'models/coco_label.txt',
		"box_score" : 0.4,
		"box_nms_iou" : 0.45,
	}

	YoloDetector.set_defaults(config)
	network = YoloDetector()
	
	fps = 0
	frame_count = 0
	start = time.time()
	while True:
		_, frame = capture.read()
		k = cv2.waitKey(1)
		if k==27 or frame is None:    # Esc key to stop
			print("End of stream.", logging.INFO)
			break
		
		network.DetectFrame(frame)
		network.DrawDetectedOnFrame(frame)


		frame_count += 1
		if frame_count >= 30:
			end = time.time()
			fps = frame_count / (end - start)
			frame_count = 0
			start = time.time()

		cv2.putText(frame, "FPS: %.2f" % fps, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		cv2.imshow("output", frame)

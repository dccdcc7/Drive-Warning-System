from enum import Enum

class LaneModelType(Enum):
	UFLD_TUSIMPLE = 0
	UFLD_CULANE = 1
	UFLDV2_TUSIMPLE = 2
	UFLDV2_CULANE = 3
	UFLDV2_CURVELANES = 4

class OffsetType(Enum):
	UNKNOWN = "确定中..."
	RIGHT = "请向右靠"
	LEFT = "请向左靠"
	CENTER = "请保持当前状态行驶"

class CurvatureType(Enum):
	UNKNOWN = "确定中..."
	STRAIGHT = "保持直行"
	EASY_LEFT =  "前方小幅右弯"
	HARD_LEFT = "前方大幅右弯"
	EASY_RIGHT = "前方小幅左弯"
	HARD_RIGHT = "前方大幅左弯"

lane_colors = [(255, 0, 0),(46,139,87),(50,205,50),(0,255,255)]

class L_BSDCollisionType(Enum):
	UNKNOWN = ""
	L_NORMAL = "Normal"
	L_PROMPT = "Warning"
	L_WARNING = "Danger"


class R_BSDCollisionType(Enum):
	UNKNOWN = ""
	R_NORMAL = "Normal"
	R_PROMPT = "Warning"
	R_WARNING = "Danger"

# config = {
#     "front_collision": 0,
#     "lane_offset": 0,
#     "lane_turn": 0,
#     "back": 0,
#     "turn_dir": "mid",
#     "offset_dir": "keep",
# }
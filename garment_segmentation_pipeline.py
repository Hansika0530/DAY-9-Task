import cv2, numpy as np, mediapipe as mp, math
from pathlib import Path
from PIL import Image
PERSON = r"C:\Users\Hansika\Downloads\segment\Blackman.png"
CLOTH  = r"C:\Users\Hansika\Downloads\segment\Blackman.png"
OUTPUT = r"C:\Users\Hansika\Downloads\segment\output"

Path(OUTPUT).mkdir(exist_ok=True)

person = cv2.imread(PERSON)
cloth  = cv2.imread(CLOTH)
H, W   = person.shape[:2]

results = {}   
results["Image"] = person.copy()

print("Running human parsing...")
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch, torch.nn.functional as F

proc  = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
model.eval()

pil   = Image.fromarray(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))
out   = model(**proc(images=pil, return_tensors="pt"))
seg   = F.interpolate(out.logits, (H, W), mode="bilinear", align_corners=False)
seg   = seg.argmax(1).squeeze().numpy().astype(np.uint8)
COLORS = {
    0:(0,0,0), 1:(0,0,128), 2:(0,0,255), 3:(221,170,51),
    4:(0,85,255), 5:(0,128,0), 6:(85,85,0), 7:(85,0,85),
    8:(0,51,85), 9:(0,255,255), 10:(0,170,255), 11:(255,0,0),
    12:(170,255,85), 13:(85,255,170), 14:(255,255,0), 15:(51,170,221),
    16:(128,128,128), 17:(128,0,128)
}

parse = np.zeros((H, W, 3), np.uint8)
for lbl, col in COLORS.items():
    parse[seg == lbl] = col
results["Parse"] = parse

# garment mask (label 4 = upper clothes, 7 = dress)
garment_mask = np.isin(seg, [4, 7]).astype(np.uint8) * 255
garment_mask = cv2.dilate(garment_mask, np.ones((15,15), np.uint8), iterations=2)

results["Agnostic-Mask"] = cv2.cvtColor(garment_mask, cv2.COLOR_GRAY2BGR)
pa = parse.copy(); pa[garment_mask > 0] = 0
results["Parse-Agnostic-v3.2"] = pa

ag = person.copy()
ag[garment_mask > 0] = (cv2.GaussianBlur(person,(51,51),0) * 0.5)[garment_mask > 0]
results["Agnostic-v3.2"] = ag

print("Running pose detection...")
pose_model = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)
pr = pose_model.process(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))
lms = pr.pose_landmarks.landmark if pr.pose_landmarks else []

def pt(i): return int(lms[i].x*W), int(lms[i].y*H)
def vis(i): return lms[i].visibility > 0.3 if lms else False
op = np.zeros((H, W, 3), np.uint8)
BONES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
KPCOLORS = [(255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),
            (0,255,0),(0,255,85),(0,255,170),(0,255,255),(0,170,255),(0,85,255),(0,0,255)]
if lms:
    for a,b in BONES:
        if vis(a) and vis(b): cv2.line(op, pt(a), pt(b), (200,200,200), 3)
    for i in range(13):
        if vis(i): cv2.circle(op, pt(i), 8, KPCOLORS[i], -1)
results["Open pose"] = op
dp = np.zeros((H, W, 3), np.uint8)
if lms and pr.segmentation_mask is not None:
    body = (pr.segmentation_mask > 0.5).astype(np.uint8)
    if all(vis(i) for i in [11,12,23,24]):
        cv2.fillPoly(dp, [np.array([pt(11),pt(12),pt(24),pt(23)])], (200,100,30))
        hx = (pt(11)[0]+pt(12)[0])//2
        hy = pt(11)[1] - abs(pt(12)[0]-pt(11)[0])//2
        cv2.ellipse(dp, (hx,hy), (40,50), 0, 0, 360, (0,215,255), -1)
    for s,e,c in [(11,13,(180,80,0)),(13,15,(180,80,0)),(12,14,(0,180,120)),(14,16,(0,180,120)),
                  (23,25,(0,100,200)),(25,27,(0,100,200)),(24,26,(50,180,50)),(26,28,(50,180,50))]:
        if vis(s) and vis(e): cv2.line(dp, pt(s), pt(e), c, 28)
    dp *= np.stack([body]*3, axis=2)
results["Dense pose"] = dp
print("Processing cloth...")
from rembg import remove as rembg_remove
cloth_pil  = Image.open(CLOTH).convert("RGBA")
cloth_nobg = np.array(rembg_remove(cloth_pil))
alpha      = cloth_nobg[:,:,3]

cloth_out  = cv2.cvtColor(cloth_nobg, cv2.COLOR_RGBA2BGR)
cloth_out[alpha < 127] = 0
results["Cloth"] = cv2.resize(cloth_out, (W, H))

_, cmask = cv2.threshold(cv2.resize(alpha,(W,H)), 127, 255, cv2.THRESH_BINARY)
results["Cloth-Mask"] = cv2.cvtColor(cmask, cv2.COLOR_GRAY2BGR)
print("Building final grid...")

ORDER  = ["Image","Agnostic-v3.2","Parse-Agnostic-v3.2","Parse","Dense pose","Open pose",
          "Cloth","Cloth-Mask","Agnostic-Mask"]
COLS   = 6
TW, TH = 200, 250   # thumbnail size
PAD    = 10
LBL_H  = 26

rows   = math.ceil(len(ORDER)/COLS)
canvas = np.full(((TH+LBL_H+PAD)*rows+PAD, (TW+PAD)*COLS+PAD, 3), 18, np.uint8)

for i, name in enumerate(ORDER):
    r, c   = i//COLS, i%COLS
    x, y   = PAD + c*(TW+PAD), PAD + r*(TH+LBL_H+PAD)
    thumb  = cv2.resize(results.get(name, np.zeros((TH,TW,3),np.uint8)), (TW,TH))
    canvas[y:y+TH, x:x+TW] = thumb
    tw, _  = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
    cv2.putText(canvas, name, (x+(TW-tw)//2, y+TH+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220,220,220), 1, cv2.LINE_AA)
out_path = str(Path(OUTPUT)/"FINAL_GRID.png")
cv2.imwrite(out_path, canvas)
print(f"\nDone! Open: {out_path}")

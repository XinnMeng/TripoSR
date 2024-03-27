import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
imgs = ["/home/xin/Dropbox/Reconstruction/TripoSR/laptop/1.jpg"]  # batch of images

# Inference
results = model(imgs)

# Results
results.print()
results.show()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0] 
import ipdb; ipdb.set_trace()
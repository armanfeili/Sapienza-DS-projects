from my_lib import *


model = VisualModelTimm(model_name="resnet50", num_classes=4, pretrained=False)

embedding = model.encode(torch.rand(8, 3, 224, 224))

print(model)
print(embedding.shape)

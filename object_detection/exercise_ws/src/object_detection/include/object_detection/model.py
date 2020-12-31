import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image


class NoGPUAvailable(Exception):
    pass

class Wrapper():
    def __init__(self, model_file):
        # TODO Instantiate your model and other class instances here!
        model = Model()
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model.model.to(device)
            model.model.load_state_dict(torch.load(model_file))
        else:
            device = torch.device('cpu')
            model.model.to(device)
            model.model.load_state_dict(torch.load(model_file),map_location=torch.device('cpu'))
        model.model.eval()
        # TODO Don't forget to set your model in evaluation/testing/production mode, and sending it to the GPU
	
        # TODO If no GPU is available, raise the NoGPUAvailable exception
        pass
    def predict(self, batch_or_image):
        # TODO Make your model predict here!

        # TODO The given batch_or_image parameter will be a numpy array (ie either a 224 x 224 x 3 image, or a
        # TODO batch_size x 224 x 224 x 3 batch of images)
        # TODO These images will be 224 x 224, but feel free to have any model, so long as it can handle these
        # TODO dimensions. You could resize the images before predicting, make your model dimension-agnostic somehow,
        # TODO etc.

        # TODO This method should return a tuple of three lists of numpy arrays. The first is the bounding boxes, the
        # TODO second is the corresponding labels, the third is the scores (the probabilities)

        # See this pseudocode for inspiration
        boxes = []
        labels = []
        scores = []
        for img in batch_or_image:  # or simply pipe the whole batch to the model instead of using a loop!
            img = Image.fromarray(img)
            box, label, score = self.model.model.predict(img) # TODO you probably need to send the image to a tensor, etc.
            box = box.numpy()
            label = label.numpy()
            score = score.numpy()
            boxes.append(box)
            labels.append(label)
            scores.append(score)

        return boxes, labels, scores

class Model():    # TODO probably extend a TF or Pytorch class!
    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)
    # TODO add your own functions if need be!
    # def forward(self, obs):
    #     return self.layers(obs)

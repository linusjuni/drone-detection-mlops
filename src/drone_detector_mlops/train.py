from drone_detector_mlops.model import Model
from drone_detector_mlops.data.data import DroneVsBirdDataset

def train():
    dataset = DroneVsBirdDataset("data/raw")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()

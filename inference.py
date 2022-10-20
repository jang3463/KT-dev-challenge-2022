import yaml
import torch
import torch.nn as nn
import ttach as tta
from utils.dataloader import CustomImageFolder, get_transform
from utils.utils import accuracy
from model.mymodel import ConvMixer
from utils.utils import seed_everything

from tqdm import tqdm


def main(model_name):

    config = yaml.load(open('./config/' + str(model_name) + '.yaml', 'r'), Loader=yaml.FullLoader)

    seed_everything(config['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    test_dataset = CustomImageFolder(root = config['test_path'],
                            transform=get_transform(train_mode='valid'))
    
    test_dataloader = torch.utils.data.DataLoader(
                            test_dataset,
                            batch_size=config['batch_size'], shuffle=True)

    transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        ])    

    model = ConvMixer(dim=config['dim'], depth=config['depth'], kernel_size=config['kernel'], patch_size=config['patch'], n_classes=config['nclasses']).to(device)
    model_state_dict = torch.load('./saved/models/best_model_0927.pt')
    model.load_state_dict(model_state_dict)

    tta_model = tta.ClassificationTTAWrapper(model, transforms, merge_mode='sum').to(device)

    accsum = 0

    tta_model.eval()

    with torch.no_grad(): 
        for X, y in tqdm(test_dataloader): 
            X, y = X.to(device), y.to(device)
            yhat = tta_model(X)
            acc = accuracy(y.cpu().data.numpy(), yhat.cpu().data.numpy().argmax(-1))
            accsum += (acc * len(y) / len(test_dataloader.dataset)) 

    print("Accuracy of test images: {:.4f}".format(accsum))

if __name__ == "__main__":
    main("mymodel")
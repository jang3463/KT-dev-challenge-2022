import yaml
import torch
import warnings
warnings.filterwarnings('ignore')

from trainer.trainer import Trainer
from utils.dataloader import CustomImageFolder, get_transform
from utils.utils import seed_everything



def main(model_name):
    config = yaml.load(open('./config/' + str(model_name) + '.yaml', 'r'), Loader=yaml.FullLoader)
    
    seed_everything(config['seed'])

    train_dataset = CustomImageFolder(root = config['train_path'],
                            transform=get_transform(train_mode=True, img_size = config['img_size']))

    valid_dataset = CustomImageFolder(root = config['valid_path'],
                            transform=get_transform(train_mode=False, img_size = config['img_size']))
    
    train_dataloader = torch.utils.data.DataLoader(
                        train_dataset, 
                        batch_size=config['batch_size'], shuffle=True)
    
    valid_dataloader = torch.utils.data.DataLoader(
                        valid_dataset,
                        batch_size=config['batch_size'], shuffle=True)

    downstream = Trainer(train_dataloader, valid_dataloader, model_name, config)
    downstream.train_step()

if __name__ == "__main__":
    main('mymodel')    
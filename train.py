import argparse
import yaml
from types import SimpleNamespace
from models.train_loop import train 

def main():
    parser = argparse.ArgumentParser(description="Motion Diffusion Model Training")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the configuration .yml file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = SimpleNamespace(**config_dict)
    
    train(config)

if __name__ == '__main__':
    main()
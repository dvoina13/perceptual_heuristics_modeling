import argparse

parser = argparse.ArgumentParser(description="args for training the neural network")
parser.add_argument("--seed", type=int, default = 1, help="seed")
parser.add_argument("--snr_choice", type=list, default = [0.65, 0.8, 1.0], help="snr_choice")
parser.add_argument("-choice", type=str, default = "only_session", help="choice")
parser.add_argument("-data", type=str, default = "M1_Phase1_Lf_gratings/", help="data directory you want to analyze")
parser.add_argument("-centering", type=bool, default = False, help="to center, or not to center, that is the question...")
parser.add_argument("-session", type=int, default = 10, help="to center, or not to center, that is the question...")

args = parser.parse_args()

seed = args.seed
snr_choice = args.snr_choice
choice = args.choice
directory = args.data
centering = args.centering
session = args.session
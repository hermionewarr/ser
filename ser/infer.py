#created 29/09/22
from pathlib import Path
import torch
import json 

from ser.data import dataloaders, test_dataloader
from ser.CNN_model import Net
from ser.constants import DATA_DIR
from ser.display import generate_ascii_art, display_num

def inference(MODEL_DIR, transforms):
	"Function to load and run a pretrained ML Model."
	print("Starting inference.")
	label = 6

    #load the parameters from the run_path so we can print them out!
	model_path = MODEL_DIR / 'model_dict'#.pt' 
	model_params = MODEL_DIR / 'parameters.json'
	print('\nModel Summary:\n')
	with open(model_params, "r") as f:
		data = json.load(f)
		for key, i in data.items():
			print(key ,': ', i)

 	# select image to run inference for
	print('\nLoading data...')
	dataloader = test_dataloader(DATA_DIR, 1, transforms)
	images, labels = next(iter(dataloader))
	while labels[0].item() != label:
		images, labels = next(iter(dataloader))

	print("Loading model...")
	model = Net()
	model.load_state_dict(torch.load(model_path))

	model.eval()
	output = model(images)
	pred = output.argmax(dim=1, keepdim=True)[0].item()
	certainty = 100*max(list(torch.exp(output)[0]))
	pixels = images[0][0]
	print(generate_ascii_art(pixels))
	print(f"This is a {pred} with certainty {certainty:.2f} %")
	display_num(pixels)
	return


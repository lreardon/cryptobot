import websocket
import json
import linecache
import baseconvert
from config import *
from fastai.vision.all import *
from fastai.vision.core import to_image
from fastai.data.all import *
from fastai.data.external import *
from PIL import Image
import torch

percentage_target = 2

polygon_socket = "wss://socket.polygon.io/stocks"
finnhub_socket = "wss://ws.finnhub.io?token="+FINNHUB_API_KEY


def collect_data():

	def on_open(ws): 
	#	print("Opened...")
	    ws.send('{"type":"subscribe","symbol":"BINANCE:BTCUSDT"}')

	def on_message(ws, message):
		global tradebook
		message = json.loads(message)
		points = message["data"]
		for point in points:
			tradebook.write(str(point)+"\n")	
			print(point)	

	on_close = lambda ws: print("Closed...")

	websocket.enableTrace(True)
	ws = websocket.WebSocketApp(finnhub_socket,
		on_open=on_open, on_message=on_message, on_close=on_close)

	with open('tradebook.jsons', 'w+') as tradebook:
		ws.run_forever()


def pre_process(source='./data/kaggle/btcusd.csv', snapshot_length=180):

	def kaggle_csv_line_to_json(line):
		line_without_break = line[:-1]
		line_as_list_of_strings = line_without_break.split(",")
		line_as_list = [float(entry) for entry in line_as_list_of_strings]
		line_headers = ['t', 'o', 'c', 'h', 'l', 'v']
		return dict(zip(line_headers, line_as_list))


	def snapshot_to_array(snapshot):
		snapshot_as_nested_array = []
		for row in snapshot:
			row_values_as_list = list(row.values())
			snapshot_as_nested_array.append(row_values_as_list)
		return snapshot_as_nested_array



	with open(source) as file:
		snapshot = []
		line_number = 0
		for line in file:
			line_number += 1
			if line_number != 1:
				line_as_dict = kaggle_csv_line_to_json(line)
				timestamp = line_as_dict['t']
#				timestamp_256 = baseconvert.base(timestamp, 10, 256)
#				number_of_digits = len(timestamp_256[:-1])
#				for power in range(number_of_digits):
#					index = number_of_digits - power - 1
#					line_as_dict['t_'+str(index)] = timestamp_256[power]
#				print(line_as_dict)
				snapshot.append(line_as_dict)
				if len(snapshot) == snapshot_length:
					# Label the "image", and store it.
					# This is the most important part of the preprocessing
					line_high = float(line_as_dict['h'])

					target_line_number = line_number + 30
					target_line = linecache.getline(source, target_line_number)
					target_line_as_dict = kaggle_csv_line_to_json(target_line)
					target_line_low = float(target_line_as_dict['l'])

					success_label = target_line_low > (line_high * (1+ .01 * percentage_target))

					# Turn snapshot into an "array" or "dataframe" that works with core.to_image
					snapshot_as_array = snapshot_to_array(snapshot)
					snapshot_as_tensor = torch.tensor(snapshot_as_array)

					training_data_location = 'data/training-data/kaggle/btcusd/'
					if success_label:
						filename = 'Snapshot'+str(timestamp)+'.pt'
					else:
						filename = 'snapshot'+str(timestamp)+'.pt'
					tensor_full_path = training_data_location+filename

					torch.save(snapshot_as_tensor, tensor_full_path)

					print(snapshot_as_tensor)
#					print(snapshot_as_tensor.shape, success_label)

#					snapshot_as_pil = to_image(snapshot_as_tensor)
#					print(snapshot_as_pil)

					# Here, bump the oldest minute out of the snapshot and add in the new one
					snapshot.pop(0)
				elif len(snapshot) > snapshot_length:
					return "ERROR - snapshot got too big!"


def build_data_block():
	def get_pt_files(path): return get_files(path, extensions='.pt')
	def label_func(fname): return "buy" if fname.name[0].isupper() else "sell"
	dblock = DataBlock(
		blocks = (TensorBlock, CategoryBlock),	# For price prediction, can change CategoryBlock to something like a scalar block,
		get_items = get_pt_files,
		get_y 	  = label_func,
		splitter  = RandomSplitter())
	return dblock

	



#------------------------- FOR FEEDING TENSORS INTO DATA_BLOCK ---------------------------------#


class TensorLoad(Transform):
    def __init__(self):
        pass

    def encodes(self, o):
        return torch.stack([torch.load(o),torch.load(o),torch.load(o)],dim=0)

def TensorBlock():
    return TransformBlock(type_tfms=TensorLoad, batch_tfms=IntToFloatTensor)


#def TOLHCV_model():


#------------------------------------------------------------------------------------------#

dls = build_data_block().dataloaders('data/training-data/kaggle/btcusd')

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
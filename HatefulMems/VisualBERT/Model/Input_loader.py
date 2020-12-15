from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import json


data = []
with open("/Volumes/Coding/HM_caompettion/Our_Own_Code/Data/train.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
       data.append(json.loads(line))

print(data[1])
print(data[1]['img'])


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = data[1]['text'] #"Replace me by any text you'd like."
print(text)
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print("waiting here")
print(output)

# image = io.imread("/Volumes/Coding/HM_caompettion/Our_Own_Code/Data/img/01245.png")

# print(type(image))
 
class Hmemes_data(Dataset):

	def __init__(self, Jsonl_type, root_dir, transform=None):

		data = []
		address = root_dir + Jsonl_type

		with open(address, 'r', encoding='utf-8') as f:
			for line in f:
				data.append(json.loads(line))

		self.allinfo   = data
		self.root_dir  = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.landmarks_frame)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		#image data
		image_addres = self.root_dir+"Data"+self.allinfo[idx]['img']
		image = io.imread(image_addres)
		#text data
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		model = BertModel.from_pretrained("bert-base-uncased")
		text = self.allinfo[idx]['text'] #"Replace me by any text you'd like."
		encoded_input = tokenizer(text, return_tensors='pt')
		txt_embedding = model(**encoded_input)

		sample = {'image': image, 'text': txt_embedding }
		return sample






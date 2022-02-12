

import torch
from torch_geometric.data import HeteroData, Batch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import argparse
from transformers import AutoModel, AutoTokenizer, BertGenerationDecoder, BertGenerationConfig, BertTokenizer, BertModel
import torch.nn as nn

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from HGT import HGT_model
import os
from pytorch_lightning import seed_everything
from transformers.file_utils import ModelOutput

seed = 42
seed_everything(seed)

# torch.multiprocessing.set_start_method('spawn')


class ROCDataset(Dataset):
	def __init__(self, split, device_id):
		assert split in ['train','test','val']
		with open('data/%s/%s.json'%(split,split), 'r', encoding='utf-8') as f:
			self.data = f.readlines()

		for idx,d in enumerate(self.data):
			self.data[idx] = eval(d.replace('\n',''))

		self.device = torch.device("cuda:%d"%device_id if torch.cuda.is_available() else "cpu")

		self.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-base")
		self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-base").to(self.device)

		for p in self.model.parameters():
			p.requires_grad = False

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		hetero_data = HeteroData()
		words_length = len(self.data[idx]['node']['word_node'])
		texts = []
		texts.append(self.data[idx]['node']['document_node'])
		texts.extend([' '.join(i) for i in self.data[idx]['node']['sentence_node']])
		texts.extend(self.data[idx]['node']['word_node'])
		inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
		with torch.no_grad():
			embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
	

		sentence_features = embeddings[1:5,:]
		# with torch.no_grad():
		# 	sentence_features = self.sentence_position_embedding.forward(sentence_features)


		hetero_data['document'].x = embeddings[0,:].unsqueeze(0)
		hetero_data['sentence'].x = sentence_features
		hetero_data['word'].x = embeddings[5::,:]

		hetero_data['document','contains','sentence'].edge_index = torch.tensor(self.data[idx]['edge']["0"], dtype=torch.long)
		hetero_data['sentence','belongsto','document'].edge_index = torch.tensor(self.data[idx]['edge']["1"],dtype=torch.long)
		hetero_data['sentence','contains','word'].edge_index = torch.tensor(self.data[idx]['edge']["2"],dtype=torch.long)
		hetero_data['word','belongsto','sentence'].edge_index = torch.tensor(self.data[idx]['edge']["3"],dtype=torch.long)
		hetero_data['sentence','next','sentence'].edge_index = torch.tensor([[0,1,2],[1,2,3]],dtype=torch.long)
		

		if self.data[idx]['sentiment'] == 'positive':
			return hetero_data, words_length, self.data[idx]['tgt'], 2, self.data[idx]['word_importance']
		elif self.data[idx]['sentiment'] == 'neutral':
			return hetero_data, words_length, self.data[idx]['tgt'], 1, self.data[idx]['word_importance']
		elif self.data[idx]['sentiment'] == 'negative':
			return hetero_data, words_length, self.data[idx]['tgt'], 0, self.data[idx]['word_importance']
		else:
			raise IndexError('error sentiment')

	@staticmethod
	def collate_fn(batch):
		hetero_datas, words_lengths, tgts, sentiments,word_importance = list(zip(*batch))

		batch_word_importance = []
		for idx,i in enumerate(word_importance):
			assert len(i) == words_lengths[idx]
			batch_word_importance.extend(i)

		return Batch.from_data_list(hetero_datas), list(words_lengths), list(tgts), torch.tensor(sentiments), torch.tensor(batch_word_importance)


class SEHGN(pl.LightningModule):

	def __init__(self, args):
		super().__init__()
		self.args = args
		# self.hparams = args
		node_type = ['document','sentence','word']
		meta_data = (['document', 'sentence', 'word'], [('document', 'contains', 'sentence'), ('sentence', 'belongsto', 'document'), ('sentence', 'contains', 'word'), ('word', 'belongsto', 'sentence'), ('sentence','next','sentence')])


		self.encoder = HGT_model(args.hidden_size, args.hidden_size, args.graph_heads, args.graph_layers, node_type_list=node_type, metadata=meta_data)
		self.tokenizer = BertTokenizer.from_pretrained("model/bert-base-uncased")
		self.decoder = BertGenerationDecoder.from_pretrained("model/bert-base-uncased",  add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102)
		self.brige = nn.Linear(2*args.hidden_size, args.hidden_size)
		self.result_id = 0

	def forward(self, graph_batch, words_lengths, tgts, sentiments, words_importance, is_Train=True):
		# print(graph_batch)

		HGT_output = self.encoder(graph_batch.x_dict, graph_batch.edge_index_dict)
		number_of_sample = len(tgts)

		document_embedding = torch.cat((graph_batch['document'].x, HGT_output['document']), 1)
		sentence_embedding = torch.cat((graph_batch['sentence'].x, HGT_output['sentence']), 1)
		word_embedding = torch.cat((graph_batch['word'].x, HGT_output['word']), 1)


		document_embedding = self.brige(document_embedding)
		sentence_embedding = self.brige(sentence_embedding)
		word_embedding = self.brige(word_embedding)

		assert HGT_output['word'].size()[0] == sum(words_lengths)


		en1 = document_embedding.reshape(number_of_sample, 1, self.args.hidden_size).to("cuda:%d"%self.args.device_id)
		en2 = sentence_embedding.reshape(number_of_sample, 4, self.args.hidden_size).to("cuda:%d"%self.args.device_id)
		en3 = torch.FloatTensor([]).to("cuda:%d"%self.args.device_id)
		en_mask1 = torch.FloatTensor([1 for _ in range(number_of_sample)]).reshape(number_of_sample, 1).to("cuda:%d"%self.args.device_id)
		en_mask2 = torch.FloatTensor([1 for _ in range(number_of_sample * 4)]).reshape(number_of_sample, 4).to("cuda:%d"%self.args.device_id)
		en_mask3 = torch.FloatTensor([]).to("cuda:%d"%self.args.device_id)

		summ = 0
		for idx,d in enumerate(words_lengths):
			summ += d
			words_lengths[idx] = summ

		for idx in range(number_of_sample):
			if idx == 0:
				padding_length = self.args.max_word_node_num-words_lengths[idx]
				tmp = torch.cat((word_embedding[0:words_lengths[idx],:], torch.tensor([[0 for _ in range(self.args.hidden_size)] for _ in range(padding_length)]).to("cuda:%d"%self.args.device_id)),0).unsqueeze(0).to("cuda:%d"%self.args.device_id)
				tmp_pad = torch.cat((torch.tensor([1 for _ in range(self.args.max_word_node_num - padding_length)])	, torch.tensor([0 for _ in range(padding_length)])),0).unsqueeze(0).to("cuda:%d"%self.args.device_id)
			elif idx == number_of_sample - 1:
				padding_length =self.args.max_word_node_num-words_lengths[idx]+words_lengths[idx-1]
				tmp = torch.cat((word_embedding[words_lengths[idx-1]::,:], torch.tensor([[0 for _ in range(self.args.hidden_size)] for _ in range(padding_length)]).to("cuda:%d"%self.args.device_id)),0).unsqueeze(0).to("cuda:%d"%self.args.device_id)
				tmp_pad = torch.cat((torch.tensor([1 for _ in range(self.args.max_word_node_num - padding_length)])	, torch.tensor([0 for _ in range(padding_length)])),0).unsqueeze(0).to("cuda:%d"%self.args.device_id)
			else:
				padding_length = self.args.max_word_node_num-words_lengths[idx]+words_lengths[idx-1]
				tmp = torch.cat((word_embedding[words_lengths[idx-1]:words_lengths[idx],:], torch.tensor([[0 for _ in range(self.args.hidden_size)] for _ in range(padding_length)]).to("cuda:%d"%self.args.device_id)),0).unsqueeze(0).to("cuda:%d"%self.args.device_id)
				tmp_pad = torch.cat((torch.tensor([1 for _ in range(self.args.max_word_node_num - padding_length)])	, torch.tensor([0 for _ in range(padding_length)])),0).unsqueeze(0).to("cuda:%d"%self.args.device_id)
			en3 = torch.cat((en3,tmp),0)
			en_mask3 = torch.cat((en_mask3,tmp_pad),0)

		encoder_hidden_states = torch.cat((en1,en2,en3),1) # 4 * 31 * 768
		encoder_attention_mask = torch.cat((en_mask1,en_mask2,en_mask3),1) # 4 * 31
		

		tokens = self.tokenizer(tgts, add_special_tokens=True, padding=True, return_tensors='pt')
		input_ids = tokens['input_ids'].to("cuda:%d"%self.args.device_id)
		attention_mask = tokens['attention_mask'].to("cuda:%d"%self.args.device_id)

		if is_Train:
			self.decoder.config.is_encoder_decoder = False
			decoder_output = self.decoder(
				input_ids=input_ids, # (batch_size, sequence_length)
				attention_mask=attention_mask, 
				encoder_hidden_states=encoder_hidden_states, # (batch_size, sequence_length, hidden_size) 
				encoder_attention_mask=encoder_attention_mask, # (batch_size, sequence_length) 
				labels=input_ids # (batch_size, sequence_length)
			)
			loss_CLM = decoder_output.loss
			
			overall_loss = loss_CLM
			return overall_loss
		
		else:
			input_ids = torch.LongTensor([[101] for _ in range(number_of_sample)]).to("cuda:%d"%self.args.device_id)

			encoder_outputs = ModelOutput()
			encoder_outputs['last_hidden_state'] = encoder_hidden_states
			encoder_outputs.last_hidden_state = encoder_hidden_states


			beam_output = self.decoder.generate(
				decoder_input_ids = input_ids,
				max_length = 24,
				num_beams = 10,
				encoder_outputs = encoder_outputs,
				encoder_hidden_states = encoder_hidden_states,
				encoder_attention_mask = encoder_attention_mask
				# is_encoder_decoder = True,
			)
			return self.tokenizer.batch_decode(beam_output, skip_special_tokens=True)


	def training_step(self, batch, batch_nb):
		loss = self.forward(*batch, is_Train=True)
		return loss

	def validation_step(self, batch, batch_nb):
		generated_endings = self.forward(*batch, is_Train=False)
		return {'generate':generated_endings}

	def test_step(self, batch, batch_nb):
		return self.validation_step(batch, batch_nb)

	def test_epoch_end(self, outputs):
		result = self.validation_epoch_end(outputs)

	def validation_epoch_end(self, outputs):
		endings = []
		for item in outputs:
			endings.extend(item['generate'])

		with open(self.args.save_dir + self.args.save_prefix + '/generated_story%d.txt'%self.result_id, 'w', encoding='utf-8') as f:
			for ending in endings:
				f.write(str(ending).lower()+'\n')
		

		self.result_id += 1


	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
		num_steps = self.args.dataset_size * self.args.epochs / self.args.grad_accum / self.args.batch_size
		scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup, num_training_steps=num_steps
        )
		return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
		
	def prepare_data(self):
		None

	def _get_dataloader(self, split_name, is_train, batch_size):
		dataset = ROCDataset(split=split_name, device_id=self.args.device_id)
		return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=0, collate_fn=ROCDataset.collate_fn)


	def train_dataloader(self):
		return self._get_dataloader('train', True, self.args.batch_size)


	def val_dataloader(self):
		return self._get_dataloader('val', False, self.args.batch_size)

	def test_dataloader(self):
		return self._get_dataloader('test', False, self.args.batch_size)




def main(args):
	checkpoint_callback = ModelCheckpoint(
		dirpath=os.path.join(args.save_dir, args.save_prefix, "checkpoints"),
		save_top_k=0,
		verbose=True,
		monitor='loss',
		mode='min',
		# save_weights_only=True,
		save_last=True,
	)
	
	print(args)
	model = SEHGN(args)
	# model.load_state_dict(torch.load('output/ESHGN_SimCSE/checkpoints/last.ckpt', map_location='cpu')['state_dict'])

	args.dataset_size = 89999

	trainer = pl.Trainer(
		gpus=[args.device_id], 
		track_grad_norm=-1,
		max_epochs=args.epochs,
		replace_sampler_ddp=False,
		accumulate_grad_batches=args.grad_accum,
		val_check_interval=0.25, # 1 for debug, 0.25 for training
		check_val_every_n_epoch=1,
		callbacks=checkpoint_callback,
		# resume_from_checkpoint = 'output/ESHGN/checkpoints/last.ckpt'
	)
	if not args.test:
		trainer.fit(model)
	else:
		trainer.test(model)




def add_model_specific_args(parser):
	parser.add_argument("--save_dir", type=str, default='output/')
	parser.add_argument("--save_prefix", type=str, default='SHGN_SimCSE_baseline')
	parser.add_argument("--lr", type=float, default=5e-5, help="Maximum learning rate")
	parser.add_argument("--graph_layers", type=int, default=1, help="GNN layers")
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--grad_accum", type=int, default=2, help="number of gradient accumulation steps")
	parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
	parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
	parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
	parser.add_argument("--debug", action='store_true', help="debug run")
	parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")
	parser.add_argument("--hidden_size", type=int, default=768, help="the hidden size of SimCSE and HGT")
	parser.add_argument("--graph_heads", type=int, default=8, help="Multi-head of Heterogeneous Graph Transformer")
	parser.add_argument("--device_id", type=int, default=0)
	parser.add_argument("--max_word_node_num", type=int, default=26)
	parser.add_argument("--test", action='store_true', help="Test only, no training")

	return parser


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="model.py")
	parser = add_model_specific_args(parser)
	args = parser.parse_args()
	path = args.save_dir + args.save_prefix
	if not os.path.exists(path):
		os.makedirs(path)
	main(args)

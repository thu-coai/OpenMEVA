import tensorflow as tf
import os
import time
import copy
import numpy as np
np.random.seed(520)
# import ipdb
import sys
import json
from eva.utils import tokenize, pro_emb
_START_VOCAB = ['<pad>', '<unk>', '<eos>']
pad_id, unk_id, eos_id = 0, 1, 2
class ruber_unrefer_model(object):
	def __init__(self,
				vocab_size,
				embed_dim,
				hidden_dim,
				learning_rate=1e-4,
				embed=None,
				loss_type="hinge_loss",
				hinge_margin=0.5):
		self.context = tf.placeholder(tf.int32, [None, None])
		self.context_length = tf.placeholder(tf.int32, [None])
		self.reference = tf.placeholder(tf.int32, [None, None])
		self.reference_length = tf.placeholder(tf.int32, [None])
		self.label = tf.placeholder(tf.int32, [None])
		self.hidden_dim = hidden_dim
		self.loss_type = loss_type
		self.hinge_margin = hinge_margin
		if embed is None:
			embed = tf.get_variable('embed', [vocab_size, embed_dim], tf.float32)
		else:
			embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)

		context_input = tf.nn.embedding_lookup(embed, self.context)
		reference_input = tf.nn.embedding_lookup(embed, self.reference)

		# batch_size = np.shape(self.context)[0]

		with tf.variable_scope("encoder_context"):
			fwbw_outputs_context, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=tf.nn.rnn_cell.GRUCell(num_units=hidden_dim), cell_bw=tf.nn.rnn_cell.GRUCell(num_units=hidden_dim), inputs=context_input, sequence_length=self.context_length, dtype=tf.float32)
			context_output = tf.concat([fwbw_outputs_context[0][:,-1,:], fwbw_outputs_context[0][:,0,:]], 1)
		with tf.variable_scope("encoder_reference"):#, reuse=tf.AUTO_REUSE):
			fwbw_outputs_reference, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=tf.nn.rnn_cell.GRUCell(num_units=hidden_dim), cell_bw=tf.nn.rnn_cell.GRUCell(num_units=hidden_dim), inputs=reference_input, sequence_length=self.reference_length, dtype=tf.float32)
			reference_output = tf.concat([fwbw_outputs_reference[0][:,-1,:], fwbw_outputs_reference[0][:,0,:]], 1)
		with tf.variable_scope("loss"):
			self.model_loss, self.accuracy, self.prob = self.get_loss(q=context_output, r=reference_output, l=self.label)
			self.epoch = tf.Variable(0, trainable=False, name='epoch')
			self.epoch_add_op = self.epoch.assign(self.epoch + 1)
			# initialize the training process
			self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
					dtype=tf.float32, name="learning_rate")
			self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * 0.95)
			self.global_step = tf.Variable(0, trainable=False, name="global_step")

		self.params = tf.trainable_variables()

		# opt = tf.train.GradientDescentOptimizer(learning_rate)
		opt = tf.train.AdamOptimizer(learning_rate)
		gradients = tf.gradients(self.model_loss, self.params)
		max_gradient_norm = 5
		clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
		self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)

		self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
				max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

	def get_loss(self, q, r, l):
		'''
			q: query or context; [batch_size, 2*hidden_dim]
			r: response or reference; [batch_size, 2*hidden_dim]
			l: label; [batch_size]
		'''
		M = tf.get_variable(
			"M",
			shape=[1, self.hidden_dim*2, self.hidden_dim*2],
			initializer=tf.truncated_normal_initializer(stddev=0.02))
		qmr = tf.reshape(tf.matmul(tf.matmul(tf.expand_dims(q, 1), M), tf.expand_dims(r, 2)), [-1, 1])
		output = tf.layers.dense(tf.concat([tf.concat([q, qmr], 1), r], 1), self.hidden_dim, name="mlp", activation=tf.tanh)
		logits = tf.layers.dense(output, 2, name="prediction")
		prob = tf.nn.softmax(logits)
		log_probs = tf.nn.log_softmax(logits, axis=-1)
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(log_probs, 1), tf.int32), tf.cast(l, tf.int32)), tf.float32))
		if self.loss_type == "cross_entropy":
			one_hot_labels = tf.one_hot(tf.cast(l, tf.int32), depth=2, dtype=tf.float32)
			per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
			loss = tf.reduce_mean(per_example_loss)
		elif self.loss_type == "hinge_loss":
			per_example_loss = tf.reduce_sum(tf.maximum((-2*tf.cast(l, tf.float32)+1)*prob[:, 1]+self.hinge_margin, tf.zeros_like(l, dtype=tf.float32)), axis=-1)
			loss = tf.reduce_mean(per_example_loss)
		else:
			raise Exception("loss_type must be 'cross_entropy' of 'hinge_loss'.")
		return loss, accuracy, prob

	def print_parameters(self):
		for item in self.params:
			print('%s: %s' % (item.name, item.get_shape()))
	
	def step_decoder(self, sess, data, forward_only=False):
		input_feed = {self.context: data['context'], 
					self.context_length: data['context_length'],
					self.reference: data['reference'],
					self.reference_length: data['reference_length'],
					self.label: data['label'],}
		if forward_only:
			output_feed = [self.model_loss, self.accuracy, self.prob]
		else:
			output_feed = [self.model_loss, self.accuracy, self.prob, self.gradient_norm, self.update]
		return sess.run(output_feed, input_feed)

def construct_negative_data(data):
	# :param data: [{'context':xxx,'model_response':xxx,'gold_response':[xxx,]},]
	neg_data = np.random.permutation(data)
	new_data = []
	for d, nd in zip(data, neg_data):
		flag = True
		for _ in range(100):
			if nd["reference"] == d["reference"]:
				nd = np.random.choice(neg_data)
			else:
				flag = False
				break
		if flag:
			raise Exception("Too little samples for negative samples construction.")
		new_data.append({"context":d["context"], "reference": nd["reference"], "label":0})
	return new_data

def load_data(data, tokenizer, max_sequence_length):
	print("begin loading data ....")
	new_data = []
	for i, tmp_data in enumerate(data):
		if not isinstance(tmp_data["reference"], list):
			tmp_data["reference"] = [tmp_data["reference"]]
		for r in tmp_data["reference"]:
			new_data.append({})
			new_data[-1]["label"] = 1
			new_data[-1]["context"] = tokenize(tmp_data["context"], tokenizer, max_sequence_length=max_sequence_length)
			new_data[-1]["reference"] = tokenize(r, tokenizer, max_sequence_length=max_sequence_length)
	np.random.shuffle(new_data)
	negative_data = construct_negative_data(new_data)
	data_segnum = len(new_data) / 20.
	data_train = np.random.permutation(new_data[:int(data_segnum*18)] + negative_data[:int(data_segnum*18)])
	data_dev = np.random.permutation(new_data[int(data_segnum*18):int(data_segnum*19)] + negative_data[int(data_segnum*18):int(data_segnum*19)])
	data_test = np.random.permutation(new_data[int(data_segnum*19):] + negative_data[int(data_segnum*19):])

	print("data_train:", len(data_train), "data_dev:", len(data_dev), "data_test:", len(data_test))
	return data_train, data_dev, data_test

def data2idx(data, vocab_list):
	word2idx = {}
	for i, v in enumerate(vocab_list):
		word2idx[v] = i
	def sent2idx(word_list):
		return [word2idx[word] if word in word2idx else unk_id for word in word_list]

	idx_data = []
	for tmp_data in data:
		idx_data.append({"reference": sent2idx(tmp_data["reference"]), 
							"context": sent2idx(tmp_data["context"]),
							"label": tmp_data["label"] if "label" in tmp_data else 0.})
	return idx_data

def build_vocab(data, vocab_size, embedding_file=None, embed_dim=300):
	print("Creating vocabulary...")
	vocab_num = {}
	for i, pair in enumerate(data):
		if i % 100000 == 0:
			print("processing line %d" % i)
		word_list = [word for word in pair["context"]] + \
				[word for word in pair["reference"]]

		for token in word_list:
			if token in vocab_num:
				vocab_num[token] += 1
			else:
				vocab_num[token] = 1
	vocab_list = _START_VOCAB + sorted(vocab_num, key=vocab_num.get, reverse=True)

	if len(vocab_list) > vocab_size:
		vocab_list = vocab_list[:vocab_size]
	else:
		vocab_size = len(vocab_list)

	with open("vocab.txt", "w") as fout:
		for word in vocab_list:
			if word in vocab_num:
				fout.write("%s,%d\n"%(word, vocab_num[word]))
			else:
				fout.write("%s\n"%(word))

	if embedding_file is None:
		embed = {}
	else:
		embed, embed_dim = pro_emb(embedding_file)

	vocab_embed = []
	for word in vocab_list:
		if word in embed:
			vector = embed[word]
		else:
			vector = np.zeros((embed_dim), dtype=np.float32)
		vocab_embed.append(vector)
	vocab_embed = np.array(vocab_embed, dtype=np.float32)

	return vocab_num, vocab_list, vocab_embed, vocab_size

def gen_batched_data(data):
	def padding(sent, l):
		return sent + [eos_id] + [pad_id] * (l-len(sent)-1)
	max_context_length = max([len(item["context"]) for item in data]) + 1
	max_reference_length = max([len(item["reference"]) for item in data]) + 1
	context, context_length, reference, reference_length, label = [], [], [], [], []
	for item in data:
		context.append(padding(item["context"], max_context_length))
		reference.append(padding(item["reference"], max_reference_length))
		context_length.append(len(item["context"])+1)
		reference_length.append(len(item["reference"])+1)
		label.append(item["label"] if "label" in item else 0.)
	batched_data = {
		"context": np.array(context),
		"context_length": np.array(context_length),
		"reference": np.array(reference),
		"reference_length": np.array(reference_length),
		"label": label
	}
	return batched_data

def train(model, sess, dataset, is_train=True, batch_size=32):
	st, ed, loss, acc = 0, 0, [], []
	while ed < len(dataset):
		st, ed = ed, ed + batch_size if ed + batch_size < len(dataset) else len(dataset)
		batch_data = gen_batched_data(dataset[st:ed])
		outputs = model.step_decoder(sess, batch_data, forward_only=False if is_train else True)
		loss.append(outputs[0])
		acc.append(outputs[1])
	if is_train:
		sess.run(model.epoch_add_op)
	return np.mean(loss), np.mean(acc)

def train_ruber_unrefer(corpus, train_dir, tokenizer, embedding_file=None, max_sequence_length=500, batch_size=32, vocab_size=40000, embed_dim=300, hidden_dim=768, loss_type="hinge_loss", hinge_margin=0.5, max_epochs=50):
	data_train, data_dev, data_test = load_data(corpus, tokenizer=tokenizer, max_sequence_length=max_sequence_length)
	vocab_num, vocab_list, vocab_embed, vocab_size = build_vocab(data_train, vocab_size=vocab_size, embedding_file=embedding_file, embed_dim=embed_dim)
	data_train, data_dev, data_test = data2idx(data_train, vocab_list), data2idx(data_dev, vocab_list), data2idx(data_test, vocab_list)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:    
		model = ruber_unrefer_model(
			vocab_size=vocab_size,
			embed_dim=embed_dim,
			hidden_dim=hidden_dim,
			embed=vocab_embed,
			loss_type=loss_type,
			hinge_margin=hinge_margin)
		model.print_parameters()

		if not os.path.exists(train_dir):
			os.mkdir(train_dir)

		with open("%s/hparam.json"%train_dir, "w") as fout:
			hparam = {"vocab_list": vocab_list, "vocab_size":vocab_size, "embed_dim": embed_dim, "hidden_dim": hidden_dim, "max_sequence_length": max_sequence_length, "tokenizer": tokenizer.name()}
			json.dump(hparam, fout, indent=4)

		if tf.train.get_checkpoint_state(train_dir):
			print("Reading model parameters from %s" % (train_dir))
			model.saver.restore(sess, tf.train.latest_checkpoint(train_dir))
		else:
			print("Created model with fresh parameters.")
			sess.run(tf.global_variables_initializer())
 
		pre_losses, best_loss = [1e18] * 3, 1e18
		for epoch in range(max_epochs):
			gen_epoch = model.epoch.eval()
			start_time = time.time()
			loss, acc = train(model, sess, data_train, batch_size=batch_size)
			if loss > max(pre_losses):
				sess.run(model.learning_rate_decay_op)
			pre_losses = pre_losses[1:] + [loss]
			print("Gen epoch %d learning rate %.4f epoch-time %.4f: loss=%.4f acc=%.4f" % (gen_epoch, model.learning_rate.eval(), time.time() - start_time, loss, acc))
			loss, acc = train(model, sess, data_dev, is_train=False, batch_size=batch_size)
			print("        dev_set loss=%.4f acc=%.4f"%(loss, acc))
			if loss < best_loss:
				best_loss = loss
				loss, acc = train(model, sess, data_test, is_train=False, batch_size=batch_size)
				print("        test_set loss=%.4f acc=%.4f"%(loss, acc))
				model.saver.save(sess, '%s/checkpoint' % train_dir, global_step=model.global_step.eval())
				print("saving parameters in %s" % train_dir)


if __name__ == "__main__":
    embedding_file = "/home/guanjian/glove/glove.6B.300d.txt" # replace the address with yours
    # you can change the function `pro_emb` in eva.utils.py to process the embedding file

    data = [
            {
                'context': "Jian is a student.",
                'reference': ["Jian comes from Tsinghua University. Jian is sleeping."],
                'candidate': "He comes from Beijing. He is sleeping.",
                'model_name': "human",
                'score': [5, 5, 5],
                'metric_score': {},
            },
            {
                'context': "Jian is a worker.",
                'reference': ["Jian came from China. Jian was running."],
                'candidate': "He came from China.",
                'model_name': "human",
                'score': [4, 4, 4],
                'metric_score': {},
            }
        ]
    from eva.tokenizer import SimpleTokenizer, PretrainedTokenizer
    tokenizer = SimpleTokenizer(method="nltk")

    from eva.model.run_ruber_unrefer import train_ruber_unrefer
    train_ruber_unrefer(data, train_dir=sys.argv[1], embedding_file=embedding_file, tokenizer=tokenizer)

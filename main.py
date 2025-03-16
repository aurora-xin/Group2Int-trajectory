import argparse
from trainer import train as GroupInt


def parse_config():
	parser = argparse.ArgumentParser()
	parser.add_argument("--cuda", default=True)
	parser.add_argument('--gpu', type=int, default=0, help='Specify which GPU to use.')
	parser.add_argument("--learning_rate", type=int, default=0.001)
	parser.add_argument("--max_epochs", type=int, default=100)
	parser.add_argument('--cfg', default='my_model')
	parser.add_argument('--train', type=int, default=0, help='Whether train or evaluate.')
	parser.add_argument("--info", type=str, default='exp', help='Name of the experiment.')
	return parser.parse_args()


def main(config):
	t = GroupInt.Trainer(config)
	if config.train==1:
		t.fit()
	else:
		t.test()	


if __name__ == "__main__":
	config = parse_config()
	main(config)

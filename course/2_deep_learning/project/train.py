import argparse
from model_object import Trainer

def train_nn(feat_dir, arch='vgg16', use_gpu=False, hidden_units=4096, epochs=1, lr=1e-4, check_point=r'./checkpoint.pht'):
    t = Trainer(feat_dir, arch, hidden_units, lr, device='cuda' if use_gpu else 'cpu')
    t.train(epochs)
    t.test_model()
    t.save_checkpoint(check_point)

   # main
parser = argparse.ArgumentParser(description='Train a NN of chosen architecture to classify 101 categories of flowers.\n')
parser.add_argument('feat_dir', 
                    type=str, 
                    help='source folder containing the training, test and validaiton set')

# options
parser.add_argument('--gpu', action = 'store_true', 
                    default=False,
                    help='use gpu for training')
parser.add_argument('--save_dir', 
                    type=str, 
                    default=r'./checkpoint.pht', 
                    dest='save_dir',
                    help='destination folder where to save the trained model')
parser.add_argument('--arch', 
                    type=str, 
                    default='vgg16', 
                    dest='architecture',
                    help='model architecture')
parser.add_argument('--hidden_units', 
                    type=int, 
                    default=4096, 
                    dest='hidden_units',
                    help='number of hidden units')
parser.add_argument('--learning_rate', 
                    type=float, 
                    default=0.0001, 
                    dest='lr',
                    help='learning rate')
parser.add_argument('--epochs', 
                    type=int, 
                    default=1, 
                    dest='epochs',
                    help='epochs to train the model for')


args = parser.parse_args()
print('---------Parameters----------')
print('gpu              = {!r}'.format(args.gpu))
print('epoch(s)         = {!r}'.format(args.epochs))
print('arch             = {!r}'.format(args.architecture))
print('learning_rate    = {!r}'.format(args.lr))
print('hidden_units     = {!r}'.format(args.hidden_units))
print('-----------------------------')


train_nn(feat_dir=args.feat_dir, 
         arch=args.architecture, 
         use_gpu=args.gpu,
         hidden_units=args.hidden_units, 
         epochs=args.epochs, 
         lr=args.lr, 
         check_point=args.save_dir)


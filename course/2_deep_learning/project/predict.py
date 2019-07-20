import argparse
import json
from model_object import Predictor

def predict(im_dir, checkpt_dir, top_k, names_map, use_gpu):
    p = Predictor(checkpt_dir)
    top_p, top_cat = p.inference(im_dir, top_k, use_gpu)
    
    if names_map:
        with open(names_map) as fn:
            map = json.load(fn)
        names = [map[str(tc)] for tc in top_cat]
    else:
        names = top_cat
    print('\n')
    print('-------Top Categories--------')
    print('categories:    {}'.format(names))
    print('probabilities: {}'.format([str(tp*100)[:4]+'%' for tp in top_p]))
    

# main
parser = argparse.ArgumentParser(description='Predict the n-likeliest categories of an image against 101 categories of flowers.\n')
parser.add_argument('image_dir', 
                    type=str, 
                    help='source folder containing the image to infere')
parser.add_argument('checkpoint', 
                    type=str, 
                    help='source folder containing the checkpointed model')

# options
parser.add_argument('--top_k', 
                    type=int, 
                    default=1, 
                    dest='top_k',
                    help='number of likeliest categories to show')
parser.add_argument('--to_names',
                    type=str, 
                    default=None, 
                    dest='json_path',
                    help='json file which links categories to specie names')
parser.add_argument('--gpu', action = 'store_true', 
                    default=False,
                    help='use gpu for training')

args = parser.parse_args()
print('---------Parameters----------')
print('image_dir        = {!r}'.format(args.image_dir))
print('checkpoint       = {!r}'.format(args.checkpoint))
print('top_k(s)         = {!r}'.format(args.top_k))
print('to_names         = {!r}'.format(args.json_path))
print('gpu              = {!r}'.format(args.gpu))
print('-----------------------------')

predict(args.image_dir, args.checkpoint, args.top_k, args.json_path, args.gpu)

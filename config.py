import argparse
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--debug', required=False, default=False, help='debug mode')
parser.add_argument('--seed', type=int, default=42, help='manual seed')
parser.add_argument('--cuda', type=int, default=0, help='cuda device')
parser.add_argument('--dataroot', required=True, help='dataset path')
parser.add_argument('--model', type=int, default=50, required=True,
                    help='resnet50 or resnet100 backbone')
parser.add_argument('--outputFolder', help=argparse.SUPPRESS)
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--augment',
                    action='store_true',
                    default=False,
                    help='augment the dataset')
parser.add_argument('--earlystop',
                    action='store_true',
                    default=False,
                    help='early stopping')


opt = parser.parse_args()

stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
opt.outputFolder = os.path.join(opt.dataroot, 'experiments', stamp)
if not os.access(opt.outputFolder, os.F_OK):
    os.makedirs(opt.outputFolder)
else:
    raise Exception('Output folder already exists: {}'.format(opt.outputFolder))

print ("Experiments folder:"+opt.outputFolder)

text_file = open(os.path.join(opt.outputFolder, "options.txt"), "w")
text_file.write(str(opt))
text_file.close()
print (opt)

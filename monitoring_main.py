import os
import string
import time
import argparse

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from recognize import recognize, load_model
from modules.mapping import mapping


class FileCreateHandler(FileSystemEventHandler):
    def on_created(self, event):
        filepath = event.src_path
        filename = os.path.basename(filepath)
        print(f"{filename} created")
        if '.txt' in filename:
            if filename == 'finish.txt':
                mapping()
        elif filename == 'sample_trim.png':
            recognize(model, opt, converter)

    def on_modified(self, event):
        filepath = event.src_path
        filename = os.path.basename(filepath)

    def on_deleted(self, event):
        filepath = event.src_path
        filename = os.path.basename(filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True,
                        help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int,
                        default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True,
                        help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int,
                        default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32,
                        help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100,
                        help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true',
                        help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true',
                        help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str,
                        required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str,
                        required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str,
                        required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20,
                        help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='the size of the LSTM hidden state')

    model = None
    opt = parser.parse_args()
    converter = None

    """ vocab / character number configuration """
    if opt.sensitive:
        # same with ASTER setting (use 94 char).
        opt.character = string.printable[:-6]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    model, opt, converter = load_model(model, opt, converter)

    target_dir = "./created_data"
    event_handler = FileCreateHandler()
    observer = Observer()
    observer.schedule(
        event_handler,
        target_dir,
        recursive=True
    )
    observer.start()
    print("File monitoring start")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

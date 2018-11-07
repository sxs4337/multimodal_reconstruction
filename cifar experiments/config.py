import argparse


def config():
    """
    List of command line arguments
    :return: command line options
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataRoot', required=True,
                        help='root folder of the dataset')
    parser.add_argument("--pklRoot", required=True,
                        help='path to pickled word2vec matrix')
    parser.add_argument('--tfRecordRoot', required=False, default=None,
                        help='root folder for tfrecord')
    parser.add_argument('--imageDir', required=False, default=None,
                        help='image directory')
    parser.add_argument('--generatorDir', required=False, default=None,
                        help='generator directory')
    parser.add_argument('--pretrainedModelRoot',
                        help='pretrained model of cifarnet checkpoint path')
    parser.add_argument('--expDir', required=True,
                        help='directory where model checkpoints are stored')
    
    parser.add_argument('--batchSize', default=64, type=int,
                        help='batch size used for training')
    parser.add_argument('--learningRate', default=0.0002, type=float,
                        help='learning rate of encoder')
    parser.add_argument('--embeddingSize', default=512, type=int,
                        help='size of the embedding')
    parser.add_argument('--numClasses', default=10, type=int,
                        help='number of classes in the dataset')
    parser.add_argument('--validationExamples', default=10000, type=int,
                        help='number of validation examples')
    parser.add_argument('--testExamples', default=10000, type=int,
                        help='number of test examples')
                        
    parser.add_argument('--numIters', default=20000, type=int,
                        help='number of epochs')
    parser.add_argument('--displayIters', default=50, type=int,
                        help='display after this iteration')    
    parser.add_argument('--validIters', default=100, type=int,
                        help='iterations after which validation occurs')
    
    parser.add_argument('--wordValidation', dest='validation',
                        action='store_const', const=0,
                        help='Use word inverse')
    parser.add_argument('--imageValidation', dest='validation',
                        action='store_const', const=1)
    parser.add_argument('--margin', default=100, type=int,
                        help="contrastive loss margin")
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--eval', dest='train', action='store_false')
    parser.add_argument('--zeroshot', dest='zero_shot', action='store_true')
    parser.set_defaults(is_zeroshot=False)
    return parser.parse_args()
 

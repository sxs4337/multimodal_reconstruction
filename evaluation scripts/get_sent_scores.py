from easy_caption_eval import caption_eval
import pickle as pkl
import sys

gts = pkl.load(open(sys.argv[1],'r'))
samples = pkl.load(open(sys.argv[2],'r'))

IDs = gts.keys()

scorer = caption_eval.COCOScorer()
scorer.score(gts, samples, IDs)

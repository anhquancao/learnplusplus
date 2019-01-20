from skmultiflow.trees import HoeffdingTree
from skmultiflow.data import RandomTreeGenerator
from skmultiflow.meta import OzaBagging
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
import matplotlib as plt


# 1. Create a stream
from learn_pp import LearnPP

stream = RandomTreeGenerator(tree_random_state=0, sample_random_state=0)
# 2. Prepare for use
stream.prepare_for_use()
# 2. Instantiate the HoeffdingTree classifier
h = [
    LearnPP(random_state=1),
    HoeffdingTree(),
    #     OzaBagging(base_estimator=HoeffdingTree(), n_estimators=5, random_state=5),
    #     OzaBoosting(base_estimator=HoeffdingTree(), n_estimators=5, random_state=5)
]

# 3. Setup the evaluator

evaluator = EvaluatePrequential(pretrain_size=1000, show_plot=True, max_samples=10000,
                                metrics=['accuracy', 'kappa'], batch_size=100)
# 4. Run
# evaluator.evaluate(stream=stream, model=h, model_names=['HT', 'ObagHT', 'OBoostHT'])
evaluator.evaluate(stream=stream, model=h, model_names=['Learn++', "HT"])

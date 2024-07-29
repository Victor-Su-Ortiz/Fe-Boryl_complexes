# main_search.py
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from deephyper.evaluator import Evaluator
from gcn_mse import run_gcn_mse 
import ray


run_function = run_gcn_mse

# Ray initialization
ray.init()

# Problem definition
hp = HpProblem()
hp.add_hyperparameter([True, False], "use_batch_norm")
hp.add_hyperparameter((16, 128), "batch_size")
hp.add_hyperparameter((3, 10), "patience")
hp.add_hyperparameter((0.1, 0.9), "scheduler_factor")
hp.add_hyperparameter((16, 256), "hidden_dim1")
hp.add_hyperparameter((16, 256), "hidden_dim2")
hp.add_hyperparameter((16, 256), "hidden_dim3")
hp.add_hyperparameter((16, 256), "hidden_dim4")
hp.add_hyperparameter((16, 256), "hidden_dim5")
hp.add_hyperparameter((16, 256), "hidden_dim6")
hp.add_hyperparameter((1, 2), "num_layers1")
hp.add_hyperparameter((1, 2), "num_layers2")
hp.add_hyperparameter((1, 2), "num_layers3")
hp.add_hyperparameter((1, 2), "num_layers4")
hp.add_hyperparameter((1, 2), "num_layers5")
hp.add_hyperparameter((1, 2), "num_layers6")
hp.add_hyperparameter((1e-5, 1e-3, "log-uniform"), "lr")
hp.add_hyperparameter([200], "epochs")

# Evaluator creation
print("Creation of the Evaluator...")
num_total_gpus_on_node = 3
evaluator = Evaluator.create(
    run_function,
    method="ray",
    method_kwargs={
        "address": "auto",
        "num_cpus": num_total_gpus_on_node,
        # "num_gpus": num_total_gpus_on_node,
        "num_cpus_per_task": 1,
        # "num_gpus_per_task": 1,
        "num_workers": 3,
    }
)
print(f"Creation of the Evaluator done with {evaluator.num_workers} worker(s)")

# Search creation
print("Creation of the search instance...")
search = CBO(
    hp,
    evaluator,
)

results = search.search(max_evals=2000)

print(results)

import kfp
from kfp import dsl 

from components.preprocess_component import preprocess_op
from components.train_component import train_op
from components.evaluate_component import evaluate_op


@dsl.pipeline(name="CIFAR-10 Pipeline", description="Pipeline for preprocessing, training, and evaluating CIFAR-10 model.")
def cifar10_pipeline():
    # Preprocessing step
    preprocess_task = preprocess_op()

    # Training step (depends on preprocessing)
    train_task = train_op().after(preprocess_task)

    # Evaluation step (depends on training)
    evaluate_task = evaluate_op().after(train_task)

# Compile the pipeline
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(cifar10_pipeline, "cifar10-pipeline.yaml")

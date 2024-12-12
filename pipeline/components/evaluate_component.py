import sys
import os
from kfp import dsl  # Import the dsl module

# Add the top-level directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from model.evaluate import evaluate_model  # Use absolute import

@dsl.component(base_image="gcr.io/group-cloud-project/evaluate-image", output_component_file="evaluate_component.yaml")  # Use the dsl.component decorator
def evaluate_op():
    """
    Evaluation component for the trained model.
    """
    return evaluate_model()  # Call the function directly

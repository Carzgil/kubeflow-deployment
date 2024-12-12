import sys
import os
from kfp import dsl  # Import the dsl module

# Add the top-level directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from model.train import train_model  # Use absolute import

@dsl.component(base_image="gcr.io/group-cloud-project/train-image", output_component_file="train_component.yaml")  # Use the dsl.component decorator
def train_op():
    """
    Training component for the model.
    """
    return train_model()  # Call the function directly

from kfp import dsl
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from data.preprocessing import preprocess_data

@dsl.component(base_image="gcr.io/group-cloud-project/preprocess-image", output_component_file="preprocess_component.yaml")
def preprocess_op():
    """
    Preprocessing component for CIFAR-10 dataset.
    """
    try:
        print("Starting preprocessing...")
        result = preprocess_data()
        print("Preprocessing completed successfully")
        return result
    except Exception as e:
        print(f"Error in preprocess_op: {e}")
        raise

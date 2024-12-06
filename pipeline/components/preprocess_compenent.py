from kfp.components import create_component_from_func

def preprocess_op():
    """
    Preprocessing component for CIFAR-10 dataset.
    """
    return create_component_from_func(
        func=None,  # Placeholder
        base_image="preprocess-image",
        output_component_file="preprocess_component.yaml"
    )

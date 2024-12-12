from google.cloud import aiplatform

PROJECT_ID='group-cloud-project'
REGION='us-east1'
BUCKET = 'cifar10-pipeline-output'
URI = 'gs://cifar10-pipeline-output'


if __name__ == "__main__":
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    template_path = f'{URI}/cifar10-pipeline.yaml' 
    
    job = aiplatform.PipelineJob(
        display_name="cifar10-pipeline",
        template_path=template_path, 
        pipeline_root=URI,
        enable_caching=True,
    )
    job.submit() 
    



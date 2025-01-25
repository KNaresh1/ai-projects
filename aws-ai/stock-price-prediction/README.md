# Stock Price Prediction using Amazon SageMaker and `XGBoost` 

## Pre-Requisites

### Create or Update conda environment
1. Add dependencies in environment.yaml
2. Run setup-env.ipynb that checks and creates/updates environment
3. Select the created environment as interpreter for your project


### Create SageMaker Role
```
aws cloudformation create-stack \
    --stack-name sagemaker-role-creation \
    --template-body file://template.yaml \
    --capabilities CAPABILITY_NAMED_IAM
```

## Cleanup AWS Resources

#### Delete SageMaker Role
```
aws cloudformation delete-stack --stack-name sagemaker-role-creation
```


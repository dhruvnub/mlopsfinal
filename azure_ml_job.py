# azure_ml_job.py
# Experiment 3: Called by Jenkins to submit training job to Azure ML
# Usage: python azure_ml_job.py --client-id ... --client-secret ... etc.

import argparse, time
from azure.ai.ml import MLClient, command
from azure.identity import ClientSecretCredential


def submit(args):
    print("\n🔐 Authenticating with Azure...")
    cred = ClientSecretCredential(
        tenant_id     = args.tenant_id,
        client_id     = args.client_id,
        client_secret = args.client_secret,
    )
    ml = MLClient(
        credential          = cred,
        subscription_id     = args.subscription_id,
        resource_group_name = args.resource_group,
        workspace_name      = args.workspace,
    )
    print(f"✅ Connected → workspace: {args.workspace}")

    job = command(
        display_name    = f"jenkins-placement-{int(time.time())}",
        command         = "python train.py",
        code            = ".",
        # Use curated Azure ML environment for scikit-learn
        environment     = "AzureML-sklearn-1.5-ubuntu20.04-py38-cpu@latest",
        compute         = args.compute,
        experiment_name = args.experiment,
        environment_variables = {
            # Point MLflow inside Azure ML job to Azure ML tracking
            "MLFLOW_TRACKING_URI": "azureml://tracking"
        },
    )

    submitted = ml.jobs.create_or_update(job)
    print(f"\n🚀 Job submitted!")
    print(f"   Job name : {submitted.name}")
    print(f"   Status   : {submitted.status}")
    print(f"   View at  : https://ml.azure.com")

    # Poll until done
    print("\n⏳ Waiting for job...")
    while True:
        status = ml.jobs.get(submitted.name).status
        print(f"   Status: {status}")
        if status in ("Completed", "Finished"):
            print("\n✅ Job completed successfully!")
            break
        elif status in ("Failed", "Canceled"):
            raise RuntimeError(f"Azure ML job {status}")
        time.sleep(30)

    return submitted.name


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--client-id",       required=True)
    p.add_argument("--client-secret",   required=True)
    p.add_argument("--tenant-id",       required=True)
    p.add_argument("--subscription-id", required=True)
    p.add_argument("--resource-group",  required=True)
    p.add_argument("--workspace",       required=True)
    p.add_argument("--experiment",      default="placement-prediction")
    p.add_argument("--compute",         default="cpu-cluster")
    submit(p.parse_args())

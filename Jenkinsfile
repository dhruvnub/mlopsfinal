// Jenkinsfile — Experiment 3: Jenkins → Azure ML Training
//
// SETUP IN JENKINS (http://localhost:8080):
//   1. Install Jenkins (jenkins.io/download → Windows)
//   2. Manage Jenkins → Plugins → Install: Git, Pipeline, Credentials Binding
//   3. Manage Jenkins → Credentials → Global → Add (Kind: Secret text):
//        ID: AZURE_CLIENT_ID       ← App Registration Client ID
//        ID: AZURE_CLIENT_SECRET   ← App Registration Secret Value
//        ID: AZURE_TENANT_ID       ← Azure Tenant ID
//        ID: AZURE_SUBSCRIPTION_ID ← Azure Subscription ID
//   4. New Item → "placement-mlops" → Pipeline
//   5. Pipeline Definition → Pipeline script from SCM → Git → your repo URL
//   6. Save → Build Now
//
// HOW TO GET AZURE CREDENTIALS:
//   Azure Portal → Azure Active Directory → App Registrations → New Registration
//   Name: "jenkins-mlops" → Register
//   Copy: Application (client) ID  and  Directory (tenant) ID
//   Go to: Certificates & Secrets → New client secret → Copy the Value
//   Then: Go to ML Workspace → Access Control (IAM) → Add role assignment
//         Role: Contributor → Members: jenkins-mlops → Save

pipeline {
    agent any

    environment {
        RESOURCE_GROUP  = 'mlops-rg'
        WORKSPACE_NAME  = 'placement-workspace'
        EXPERIMENT_NAME = 'placement-prediction'
        COMPUTE_NAME    = 'cpu-cluster'
        ACR_NAME        = 'placementmlops.azurecr.io'
    }

    stages {

        stage('Checkout') {
            steps {
                echo '📥 Pulling latest code from GitHub...'
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                echo '🔧 Installing Python packages...'
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Run Unit Tests') {
            steps {
                echo '✅ Running tests before training...'
                bat 'pytest tests/ -v --tb=short'
            }
        }

        stage('Submit Azure ML Training Job') {
            steps {
                echo '☁️  Submitting training job to Azure ML...'
                withCredentials([
                    string(credentialsId: 'AZURE_CLIENT_ID',       variable: 'AZ_CLIENT_ID'),
                    string(credentialsId: 'AZURE_CLIENT_SECRET',    variable: 'AZ_CLIENT_SECRET'),
                    string(credentialsId: 'AZURE_TENANT_ID',        variable: 'AZ_TENANT_ID'),
                    string(credentialsId: 'AZURE_SUBSCRIPTION_ID',  variable: 'AZ_SUB_ID'),
                ]) {
                    bat """
                        python azure_ml_job.py ^
                            --client-id       %AZ_CLIENT_ID%    ^
                            --client-secret   %AZ_CLIENT_SECRET% ^
                            --tenant-id       %AZ_TENANT_ID%    ^
                            --subscription-id %AZ_SUB_ID%       ^
                            --resource-group  %RESOURCE_GROUP%  ^
                            --workspace       %WORKSPACE_NAME%  ^
                            --experiment      %EXPERIMENT_NAME% ^
                            --compute         %COMPUTE_NAME%
                    """
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                echo '🐳 Building Docker image...'
                bat 'docker build -t placement-api:latest .'
                echo '✅ Image built: placement-api:latest'
            }
        }

        stage('Push to Azure Container Registry') {
            steps {
                echo '📦 Pushing image to Azure Container Registry...'
                withCredentials([
                    string(credentialsId: 'AZURE_CLIENT_ID',      variable: 'AZ_CLIENT_ID'),
                    string(credentialsId: 'AZURE_CLIENT_SECRET',   variable: 'AZ_CLIENT_SECRET'),
                ]) {
                    bat """
                        docker login %ACR_NAME% -u %AZ_CLIENT_ID% -p %AZ_CLIENT_SECRET%
                        docker tag  placement-api:latest %ACR_NAME%/placement-api:latest
                        docker push %ACR_NAME%/placement-api:latest
                    """
                }
                echo "✅ Image pushed to: ${env.ACR_NAME}/placement-api:latest"
            }
        }
    }

    post {
        success {
            echo '🎉 Pipeline complete! Model trained on Azure ML and image pushed to ACR.'
            echo '   Azure ML Studio → https://ml.azure.com'
            echo '   Container Registry → Azure Portal → placementmlops'
        }
        failure {
            echo '❌ Pipeline failed. Check the stage output above.'
            echo '   For Azure ML errors → https://ml.azure.com'
        }
    }
}

# GKE CI/CD Deployment

This project ships as one production container:

- Vite builds the Vue frontend into static files.
- FastAPI serves the API, `/health`, and the compiled frontend.
- SQLite and uploaded claim photos are stored on a GKE persistent volume at `/data`.

The deployment uses one replica because SQLite and local file uploads are not safe for multi-pod writes. Move the database to Cloud SQL and uploads to Cloud Storage before increasing replicas.

## Required Google Cloud resources

Create or choose:

- A GKE cluster
- An Artifact Registry Docker repository
- A Google service account for GitHub Actions
- A Workload Identity Federation provider connected to your GitHub repository

The service account needs permissions to push Artifact Registry images and deploy to GKE, for example:

- `roles/artifactregistry.writer`
- `roles/container.developer`

## GitHub secrets

Add these repository secrets:

| Secret | Example |
| --- | --- |
| `GCP_PROJECT_ID` | `my-gcp-project` |
| `GKE_CLUSTER` | `claims-cluster` |
| `GKE_LOCATION` | `us-central1` or `us-central1-a` |
| `GAR_LOCATION` | `us-central1` |
| `GAR_REPOSITORY` | `claims-portal` |
| `WORKLOAD_IDENTITY_PROVIDER` | `projects/123456789/locations/global/workloadIdentityPools/github/providers/my-repo` |
| `GCP_SERVICE_ACCOUNT` | `github-deployer@my-gcp-project.iam.gserviceaccount.com` |

## Local Docker build

```powershell
docker build -t claims-portal:local .
docker run --rm -p 8000:8000 -v ${PWD}/.local-data:/data claims-portal:local
```

Open `http://localhost:8000`.

## Manual Kubernetes deploy

Before applying manually, replace the image in `k8s/deployment.yaml` or use:

```powershell
kubectl apply -k k8s
kubectl set image deployment/claims-portal claims-portal=REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY/claims-portal:TAG -n claims-portal
kubectl rollout status deployment/claims-portal -n claims-portal
```

Get the public URL:

```powershell
kubectl get service claims-portal -n claims-portal
```

@echo off
echo ==============================================
echo 🚀 Deploying to Vercel and GitHub 🚀
echo ==============================================
echo.
echo Committing deployment configuration files to Git...
git add render.yaml frontend/vercel.json
git commit -m "Add Render and Vercel deployment configs"

echo.
echo Pushing to GitHub (car-damage-claim-frontend branch)...
git push origin car-damage-claim-frontend

echo.
echo ==============================================
echo Starting Vercel deployment for the Frontend...
echo ==============================================
cd frontend
call npx vercel

echo.
echo ==============================================
echo Deployment commands finished!
echo Note: For the backend, please go to https://dashboard.render.com/blueprints 
echo and create a New Blueprint using your GitHub repository.
echo ==============================================
pause

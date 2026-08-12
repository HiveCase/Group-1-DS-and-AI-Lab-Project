@echo off
echo ==============================================
echo 🚀 Deploying to Vercel and GitHub 🚀
echo ==============================================
echo.
echo Committing deployment configuration files to Git...
git add .
git commit -m "Add Render configs and fix Pydantic validation"

echo.
echo Pushing to GitHub (car-damage-claim-frontend branch)...
git push origin car-damage-claim-frontend

echo.
echo ==============================================
echo Deployment commands finished!
echo Go check Render to see it deploy successfully!
echo ==============================================
pause

# Render Deployment Guide

## Quick Deployment Steps

### 1. Sign Up for Render

1. Go to [render.com](https://render.com)
2. Click **"Get Started for Free"**
3. Sign up with your **GitHub account** (recommended)

### 2. Create New Web Service

1. From your Render dashboard, click **"New +"**
2. Select **"Web Service"**
3. Click **"Connect a repository"**
4. Find and select: `energy-accuracy-tradeoff-iot-activity-recognition`
5. Click **"Connect"**

### 3. Configure Your Service

Render will auto-detect your settings, but verify these:

**Basic Settings:**

- **Name**: `energy-accuracy-tradeoff` (or your preferred name)
- **Region**: Choose closest to you (e.g., Oregon, Frankfurt, Singapore)
- **Branch**: `main`
- **Root Directory**: Leave blank

**Build & Deploy:**

- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`

**Instance Type:**

- Select **"Free"** (no credit card required)

### 4. Deploy!

1. Click **"Create Web Service"**
2. Render will start building your app (takes 2-5 minutes)
3. Watch the deployment logs in real-time
4. Once complete, you'll get a URL like: `https://energy-accuracy-tradeoff.onrender.com`

### 5. Access Your Live App

Your app will be live at the provided URL! 🎉

**Pages:**

- Home: `https://your-app.onrender.com/`
- Results: `https://your-app.onrender.com/results`
- Predict: `https://your-app.onrender.com/predict`
- Compare: `https://your-app.onrender.com/compare`

**API:**

- Health: `https://your-app.onrender.com/api/health`
- Methods: `https://your-app.onrender.com/api/methods`

---

## Important Notes

### ⚠️ Free Tier Limitations

- **Auto-sleep**: App sleeps after 15 minutes of inactivity
- **Wake-up time**: First request after sleep takes 30-60 seconds
- **Monthly hours**: 750 hours/month (plenty for demos and portfolios)

### 🔄 Automatic Deployments

Every time you push to GitHub's `main` branch, Render will:

1. Automatically detect the changes
2. Rebuild your app
3. Deploy the new version

**No manual redeployment needed!**

### 📊 Monitoring

From your Render dashboard, you can:

- View deployment logs
- Monitor app performance
- Check error logs
- Restart the service if needed

---

## Troubleshooting

### Build Fails

**Issue**: Dependencies fail to install

**Solution**: Check that `requirements.txt` is up to date

```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update requirements"
git push
```

### App Won't Start

**Issue**: Gunicorn can't find the app

**Solution**: Verify start command is exactly: `gunicorn app:app`

### Visualizations Not Loading

**Issue**: PNG files not found

**Solution**: Make sure all plot PNG files are committed to GitHub:

```bash
git add *.png
git commit -m "Add visualization plots"
git push
```

### Slow First Load

**Issue**: App takes long to respond initially

**Solution**: This is normal on free tier (auto-sleep). The app wakes up on first request.

---

## Upgrading (Optional)

If you need better performance:

1. Go to your service settings
2. Change instance type from "Free" to "Starter" ($7/month)
3. Benefits:
   - No auto-sleep
   - Faster performance
   - More resources

---

## Environment Variables (If Needed)

If you need to add environment variables:

1. Go to your service in Render
2. Click **"Environment"** tab
3. Add key-value pairs
4. Click **"Save Changes"**

Example:

- `FLASK_ENV`: `production`
- `SECRET_KEY`: `your-secret-key`

---

## Custom Domain (Optional)

To use your own domain:

1. Go to **"Settings"** → **"Custom Domain"**
2. Add your domain
3. Update DNS records as instructed
4. Render provides free SSL certificates

---

## Next Steps After Deployment

1. **Test all pages**: Visit each page and verify functionality
2. **Test API endpoints**: Use curl or Postman to test API
3. **Share your URL**: Add it to your GitHub README
4. **Monitor logs**: Check for any errors in production

---

## Support

- **Render Docs**: [docs.render.com](https://docs.render.com)
- **Community**: [community.render.com](https://community.render.com)
- **Status**: [status.render.com](https://status.render.com)

---

## Your App is Ready! 🚀

Your Flask application is now configured for Render deployment. Just follow the steps above to get it live!

**Estimated deployment time**: 5-10 minutes

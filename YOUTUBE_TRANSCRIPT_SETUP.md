# YouTube Transcript Service Setup Guide

## The Problem: IP Blocking

When you see this error:
```json
{
    "success": false,
    "error": "Unexpected error getting transcript: no element found: line 1, column 0"
}
```

This means **YouTube is blocking your IP address**. This commonly happens when:

- 🏢 Running on cloud servers (AWS, GCP, Azure, DigitalOcean, etc.)
- 🌐 Your ISP or network is flagged
- 🚫 Too many requests from the same IP
- 🔒 Age-restricted or private videos

## Solution: WebShare Proxy

The YouTube transcript service supports WebShare proxy to bypass IP blocking.

### Step 1: Check Current Proxy Status

```bash
curl http://localhost:8000/youtube_proxy_status
```

If you see `"proxy_configured": false`, you need to set up WebShare credentials.

### Step 2: Sign Up for WebShare

1. Go to **https://www.webshare.io/**
2. Create an account (they offer free trial)
3. Purchase a **RESIDENTIAL** proxy package
   - ⚠️ **Important**: Do NOT buy "Proxy Server" or "Static Residential" packages
   - ✅ Only "Residential" proxies work for YouTube

### Step 3: Get Your Credentials

1. Login to WebShare dashboard
2. Navigate to "Proxy Settings" or "API"
3. Copy your:
   - **Proxy Username** (also called API key or username)
   - **Proxy Password**

### Step 4: Set Environment Variables

Add these to your `.env` file:

```bash
WEBSHARE_PROXY_USERNAME=your-actual-username-here
WEBSHARE_PROXY_PASSWORD=your-actual-password-here
```

**Or** export them in your terminal:

```bash
export WEBSHARE_PROXY_USERNAME=your-actual-username-here
export WEBSHARE_PROXY_PASSWORD=your-actual-password-here
```

### Step 5: Restart Your Server

```bash
# Stop current server (Ctrl+C)
# Then restart:
uvicorn pipeline:app --reload
```

### Step 6: Test with Proxy Enabled

```bash
curl -X POST http://localhost:8000/extract_youtube_transcript \
  -F "video_url=https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  -F "use_proxy=true" \
  -F "languages=en"
```

You should now see:
```json
{
    "success": true,
    "proxy_status": {
        "proxy_enabled": true,
        "proxy_status": "enabled",
        "has_credentials": true
    },
    ...
}
```

## Troubleshooting

### Error: "Missing WebShare credentials"

**Check logs for:**
```
❌ WebShare proxy requested but credentials not found in environment.
   Missing environment variables: WEBSHARE_PROXY_USERNAME, WEBSHARE_PROXY_PASSWORD
```

**Solution:**
- Verify credentials are in `.env` file
- Make sure `.env` file is in the project root directory
- Ensure you restarted the server after adding credentials
- Check for typos in variable names (must match exactly)

### Error: Still getting "no element found" with proxy enabled

**Possible causes:**
1. Wrong proxy package (you need RESIDENTIAL, not Static/Proxy Server)
2. Proxy credentials expired or incorrect
3. Video is age-restricted or unavailable
4. WebShare account balance/credits depleted

**Verify:**
```bash
# Check proxy status
curl http://localhost:8000/youtube_proxy_status

# Check server logs for:
# "✅ YouTubeTranscriptService initialized with WebShare proxy"
```

### Alternative: Use WetroCloud Service

If you don't want to use proxies, use the existing WetroCloud endpoint:

```bash
curl -X POST http://localhost:8000/process_youtube \
  -F 'youtube_urls=["https://www.youtube.com/watch?v=VIDEO_ID"]' \
  -F "space_id=your-space-id"
```

This uses a different API service that doesn't require proxies.

## API Endpoints Summary

### 1. Extract Transcript (with proxy support)
```
POST /extract_youtube_transcript
- video_url: YouTube URL
- use_proxy: true/false
- languages: comma-separated (e.g., "en,es")
```

### 2. Check Proxy Status
```
GET /youtube_proxy_status
```

### 3. Get Available Transcripts
```
GET /youtube_transcript_info?video_url=URL&use_proxy=true
```

### 4. Process YouTube (WetroCloud - no proxy needed)
```
POST /process_youtube
- youtube_urls: JSON array
- space_id: your space ID
```

## Cost Considerations

- **WebShare**: Pay per proxy/bandwidth (residential proxies ~$2.99-$10/month)
- **WetroCloud**: API-based, pricing varies
- **No Proxy**: Free but only works on non-blocked IPs (local development)

## Best Practice

For **production/cloud deployments**: Always use `use_proxy=true` with WebShare
For **local development**: Try without proxy first, enable if blocked

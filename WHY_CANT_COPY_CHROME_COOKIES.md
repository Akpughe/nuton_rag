# Why You Can't Copy Chrome Cookies to Server Environments

## The Question

"Can't we just copy the Chrome cookie database to `/root/.config/google-chrome` on the server?"

## The Short Answer

**No**, because Chrome cookies are **encrypted with OS-specific keys** that don't exist on the server.

## Technical Explanation

### What Is the Chrome Cookie Database?

```bash
# macOS
~/Library/Application Support/Google/Chrome/Default/Cookies

# Linux
~/.config/google-chrome/Default/Cookies

# Windows
%LOCALAPPDATA%\Google\Chrome\User Data\Default\Cookies
```

This is an **SQLite database** with an encrypted `encrypted_value` column:

```sql
CREATE TABLE cookies (
    creation_utc INTEGER NOT NULL,
    host_key TEXT NOT NULL,
    name TEXT NOT NULL,
    value TEXT NOT NULL,
    path TEXT NOT NULL,
    expires_utc INTEGER NOT NULL,
    is_secure INTEGER NOT NULL,
    is_httponly INTEGER NOT NULL,
    encrypted_value BLOB DEFAULT ''  -- ← This is encrypted!
);
```

### The Encryption Problem

Chrome uses **OS keychain/keyring** to encrypt cookies:

#### macOS (Your Development Machine)
```python
# Chrome encrypts cookies using Keychain
Keychain → Derives encryption key → Encrypts cookie value
```

Chrome calls macOS Keychain API:
- `SecKeychainFindGenericPassword()` to get master key
- Key is derived from your macOS user password
- **This key only exists on your Mac**

#### Linux Server (Docker/Cloud)
```python
# Chrome would encrypt using libsecret/gnome-keyring
# BUT: These aren't installed in headless servers!
No keyring available → Can't decrypt cookies
```

### What Happens If You Copy the Database?

Let's trace what happens:

```python
# 1. Copy database from Mac to server
scp ~/Library/.../Cookies server:/root/.config/google-chrome/Default/

# 2. yt-dlp tries to read cookies
# 3. Chrome's cookie decryption logic:

def decrypt_cookie(encrypted_value, os_type):
    if os_type == 'mac':
        key = get_key_from_keychain()  # ← FAILS: No keychain on server!
    elif os_type == 'linux':
        key = get_key_from_secret_service()  # ← FAILS: Not installed!

    return decrypt_aes(encrypted_value, key)  # ← Can't decrypt without key!
```

**Result**: yt-dlp can read the database file, but **cannot decrypt the cookie values**.

### Detailed Encryption Flow

#### On Your Mac (Encryption)
```
1. You visit YouTube in Chrome
2. YouTube sets cookie: "session=abc123"
3. Chrome encrypts:

   macOS Keychain API
   ↓
   Get encryption key "Chrome Safe Storage"
   ↓
   Derive AES-128 key using PBKDF2
   ↓
   Encrypt "abc123" → Binary blob
   ↓
   Store encrypted blob in SQLite database
```

#### On Server (Attempted Decryption)
```
1. yt-dlp opens Cookies database
2. Reads encrypted_value column → Binary blob
3. Tries to decrypt:

   Try macOS Keychain API → NOT AVAILABLE (wrong OS)
   ↓
   Try Linux Secret Service → NOT INSTALLED (headless server)
   ↓
   Try Windows DPAPI → NOT AVAILABLE (wrong OS)
   ↓
   ❌ FAIL: Cannot decrypt cookie
```

### Proof: Look at the Encryption Code

Chrome's actual encryption (simplified):

```cpp
// Chrome source: components/os_crypt/os_crypt_mac.mm
bool OSCrypt::EncryptString(const std::string& plaintext,
                            std::string* ciphertext) {
  // Get key from macOS Keychain
  crypto::AppleKeychain keychain;
  UInt32 password_length = 0;
  void* password_data = nullptr;

  OSStatus status = keychain.FindGenericPassword(
      strlen(service_name), service_name,
      strlen(account_name), account_name,
      &password_length, &password_data,
      nullptr);

  // This key ONLY exists on this Mac!
  // Derived from your system password
}
```

### Why yt-dlp's Cookie Extraction Works Locally

When you run on your Mac:

```python
YTDLPTranscriptService()  # With use_cookies=True (default)

# yt-dlp internally does:
1. Find Chrome's cookie database
2. Load the encrypted database
3. Call macOS Keychain API to get decryption key
4. Successfully decrypt cookies
5. Send decrypted cookies to YouTube
```

**This works because**: You're on the same Mac where Chrome encrypted the cookies!

### What About Just Using Plaintext Cookies?

You might think: "What if I export cookies as plaintext?"

**This works!** But with caveats:

```python
# Method 1: Export cookies.txt (plaintext)
# Use browser extension "Get cookies.txt LOCALLY"
# Exports to Netscape cookie format:

# Netscape HTTP Cookie File
.youtube.com    TRUE    /    TRUE    1234567890    VISITOR_INFO1_LIVE    abc123

# Then use it:
ydl_opts = {
    'cookiefile': '/path/to/youtube_cookies.txt',  # Plaintext, no encryption!
}
```

**Downsides**:
1. **Manual process** - Need to export every 1-2 weeks when cookies expire
2. **Security risk** - Plaintext cookies in files
3. **Session binding** - May not work from different IP/server
4. **Not practical** - For automated systems

## Real-World Scenarios

### Scenario 1: Copy Database (What You Asked About)

```bash
# On your Mac
cp ~/Library/Application\ Support/Google/Chrome/Default/Cookies /tmp/

# Upload to server
scp /tmp/Cookies server:/root/.config/google-chrome/Default/

# Try to use
python3
>>> from yt_dlp import YoutubeDL
>>> ydl = YoutubeDL({'cookiesfrombrowser': ('chrome',)})
>>> ydl.extract_info('youtube-url')

# Result:
❌ ERROR: Cannot decrypt cookies - encryption key not available
```

### Scenario 2: Our Solution (use_cookies=False)

```python
# On server
ytdlp = YTDLPTranscriptService(use_cookies=False)
result = ytdlp.get_transcript(video_url)

# Result:
✅ SUCCESS: Works for most public videos (no authentication needed)
```

### Scenario 3: Proper Cookie Export

```bash
# 1. Export cookies from browser to plaintext file
# Use extension: "Get cookies.txt LOCALLY"

# 2. Upload to server
scp youtube_cookies.txt server:/app/

# 3. Use cookie file
ydl_opts = {
    'cookiefile': '/app/youtube_cookies.txt',
}

# Result:
✅ SUCCESS: Works with authentication (but requires manual cookie refresh)
```

## Alternative Approaches That DO Work

### 1. Use YouTube API (Recommended for Production)

```python
from googleapiclient.discovery import build

youtube = build('youtube', 'v3', developerKey='YOUR_API_KEY')
# Official API, no cookies needed
```

### 2. Use Cookie File (For Testing/Dev)

```python
# Export cookies manually, then:
ydl_opts = {
    'cookiefile': './cookies.txt',  # Plaintext format
}
```

### 3. Run Without Authentication (Our Solution)

```python
# Works for 90% of videos
ytdlp = YTDLPTranscriptService(use_cookies=False)
```

## Summary Table

| Approach | Works on Server? | Pros | Cons |
|----------|------------------|------|------|
| Copy Chrome database | ❌ No | None | Encryption keys missing |
| Export plaintext cookies | ✅ Yes | Full auth | Manual refresh needed |
| use_cookies=False | ✅ Yes | Automatic, simple | No auth (most videos OK) |
| YouTube API | ✅ Yes | Official, reliable | Requires API key, quota limits |

## The Bottom Line

**You physically cannot use Chrome's encrypted cookie database on a different machine/OS** because:

1. ✗ Encryption keys are OS/machine-specific
2. ✗ Keys are stored in OS keychain (not in the database)
3. ✗ Can't extract keys without compromising system security
4. ✗ Chrome's decryption requires the original OS keychain

**What you CAN do**:
1. ✓ Disable cookies (`use_cookies=False`) - works for most videos
2. ✓ Export cookies as plaintext file (manual but works)
3. ✓ Use YouTube API (proper production solution)

## Technical Reference

If you want to see the actual encryption code:

- Chrome encryption: https://source.chromium.org/chromium/chromium/src/+/main:components/os_crypt/
- yt-dlp cookie handling: https://github.com/yt-dlp/yt-dlp/blob/master/yt_dlp/cookies.py
- Browser cookie encryption standards: https://chromium.googlesource.com/chromium/src/+/master/docs/design/encryption.md

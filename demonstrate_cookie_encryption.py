#!/usr/bin/env python3
"""
Demonstration: Why you can't copy Chrome cookies to another machine.

This script shows the actual Chrome cookie database structure and encryption.
"""

import sqlite3
import os
from pathlib import Path


def find_chrome_cookies_db():
    """Find Chrome's Cookies database on this system."""
    possible_paths = [
        # macOS
        Path.home() / "Library/Application Support/Google/Chrome/Default/Cookies",
        Path.home() / "Library/Application Support/Google/Chrome/Default/Network/Cookies",
        # Linux
        Path.home() / ".config/google-chrome/Default/Cookies",
        # Windows
        Path(os.getenv('LOCALAPPDATA', '')) / "Google/Chrome/User Data/Default/Cookies",
        Path(os.getenv('LOCALAPPDATA', '')) / "Google/Chrome/User Data/Default/Network/Cookies",
    ]

    for path in possible_paths:
        if path.exists():
            return path
    return None


def examine_cookie_encryption():
    """Examine Chrome's cookie database to show encryption."""
    print("=" * 70)
    print("Chrome Cookie Database Examination")
    print("=" * 70)

    cookies_db = find_chrome_cookies_db()

    if not cookies_db:
        print("\n❌ Chrome cookies database not found on this system")
        print("\nThis demonstrates the first problem:")
        print("→ The database doesn't exist in server/container environments!")
        return

    print(f"\n✓ Found Chrome cookies database:")
    print(f"  {cookies_db}")

    # Check if we can read it
    if not os.access(cookies_db, os.R_OK):
        print("\n❌ Cannot read database (Chrome might be running)")
        print("\nTIP: Close Chrome and try again")
        return

    try:
        # Make a temporary copy (Chrome locks the main file)
        import tempfile
        import shutil

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            tmp_path = tmp.name

        shutil.copy2(cookies_db, tmp_path)

        # Open the database
        conn = sqlite3.connect(tmp_path)
        cursor = conn.cursor()

        # Get database schema
        print("\n" + "=" * 70)
        print("Database Schema")
        print("=" * 70)

        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='cookies'")
        schema = cursor.fetchone()
        if schema:
            print("\n" + schema[0])

        # Get a sample cookie (look for YouTube)
        print("\n" + "=" * 70)
        print("Sample YouTube Cookie (Encrypted)")
        print("=" * 70)

        cursor.execute("""
            SELECT host_key, name, value, encrypted_value, length(encrypted_value)
            FROM cookies
            WHERE host_key LIKE '%youtube%'
            LIMIT 1
        """)

        cookie = cursor.fetchone()

        if cookie:
            host, name, plaintext_value, encrypted_value, encrypted_len = cookie

            print(f"\nHost: {host}")
            print(f"Name: {name}")
            print(f"Plaintext Value: '{plaintext_value}'")
            print(f"Encrypted Value Length: {encrypted_len} bytes")

            if encrypted_value:
                # Show first 50 bytes of encrypted data
                preview = encrypted_value[:50] if len(encrypted_value) > 50 else encrypted_value
                hex_preview = preview.hex() if isinstance(preview, bytes) else str(preview)
                print(f"Encrypted Value (hex preview): {hex_preview}...")

                print("\n" + "=" * 70)
                print("⚠️  KEY INSIGHT")
                print("=" * 70)
                print("\nThe 'encrypted_value' column contains binary encrypted data.")
                print("This data is encrypted using a key from your OS keychain.")
                print("\n❌ If you copy this database to a server:")
                print("   1. The database file itself copies fine")
                print("   2. BUT the encryption key is NOT in the database")
                print("   3. The key lives in your macOS Keychain (or Linux keyring)")
                print("   4. Without the key, the encrypted_value is useless gibberish")
                print("\n✓ This is why yt-dlp can extract cookies locally:")
                print("   → It reads the database AND gets the key from your keychain")
                print("\n✗ This is why copying the database to a server fails:")
                print("   → The server has the database but NOT the keychain/key")

        else:
            print("\nNo YouTube cookies found in database")
            print("(Visit youtube.com in Chrome, then try again)")

        # Show where the encryption key is stored
        print("\n" + "=" * 70)
        print("Where Is The Encryption Key?")
        print("=" * 70)

        import platform
        os_type = platform.system()

        if os_type == 'Darwin':  # macOS
            print("\nOn macOS, the key is stored in Keychain:")
            print("  Location: ~/Library/Keychains/")
            print("  Key name: 'Chrome Safe Storage'")
            print("  Access: Requires macOS Keychain API")
            print("\n  You can see it in Keychain Access app:")
            print("  → Open 'Keychain Access'")
            print("  → Search for 'Chrome'")
            print("  → Look for 'Chrome Safe Storage'")
        elif os_type == 'Linux':
            print("\nOn Linux, the key is stored in:")
            print("  - gnome-keyring (GNOME)")
            print("  - kwallet (KDE)")
            print("  - Or encrypted with 'peanuts' as default password")
        elif os_type == 'Windows':
            print("\nOn Windows, the key is stored in:")
            print("  - DPAPI (Data Protection API)")
            print("  - Encrypted with your Windows user password")

        print("\n" + "=" * 70)
        print("Conclusion")
        print("=" * 70)
        print("\n1. Chrome cookies database = SQLite file (CAN be copied)")
        print("2. Encryption key = OS keychain/keyring (CANNOT be copied)")
        print("3. Without the key, the database is useless on another machine")
        print("\n→ This is why YTDLPTranscriptService(use_cookies=False) exists!")

        conn.close()
        os.unlink(tmp_path)

    except sqlite3.Error as e:
        print(f"\n❌ Database error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def show_yt_dlp_cookie_extraction():
    """Show how yt-dlp extracts cookies."""
    print("\n" + "=" * 70)
    print("How yt-dlp Extracts Cookies (Simplified)")
    print("=" * 70)

    print("""
When you use: YTDLPTranscriptService(browser='chrome')

yt-dlp does this:

┌─────────────────────────────────────────────────────────────┐
│ Step 1: Find Chrome's cookie database                      │
│ → Searches standard paths for Chrome/Cookies file          │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Open SQLite database                               │
│ → Reads cookies table                                       │
│ → Gets encrypted_value column                              │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Get encryption key from OS keychain                │
│ → macOS: Calls Keychain API for "Chrome Safe Storage"      │
│ → Linux: Calls Secret Service API                          │
│ → Windows: Calls DPAPI                                     │
│ ⚠️  THIS STEP FAILS ON SERVERS (no keychain installed!)   │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Decrypt cookie values                              │
│ → Uses key from Step 3 to decrypt Step 2 data              │
│ → AES-128 decryption with PBKDF2 key derivation           │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Send decrypted cookies to YouTube                  │
│ → Now authenticated as you!                                │
└─────────────────────────────────────────────────────────────┘

⚠️  On a server without Chrome/keychain:
   → Steps 1-2 FAIL (no database file exists)
   → Even if you copy the database, Step 3 FAILS (no keychain)
   → Without Step 3, Step 4 produces garbage (wrong/missing key)

✓  Solution: Use YTDLPTranscriptService(use_cookies=False)
   → Skips all cookie steps
   → Works for most public videos
""")


if __name__ == "__main__":
    print("\n🔍 Chrome Cookie Encryption Demonstration\n")

    examine_cookie_encryption()
    show_yt_dlp_cookie_extraction()

    print("\n" + "=" * 70)
    print("For more details, see: WHY_CANT_COPY_CHROME_COOKIES.md")
    print("=" * 70)

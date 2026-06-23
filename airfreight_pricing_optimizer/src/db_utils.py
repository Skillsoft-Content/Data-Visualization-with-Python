"""
Database environment diagnostics and connection error translation.

The most common failure on Windows is the Microsoft Store Python alias, which
sandboxes registry and DLL access that pyodbc needs to find ODBC drivers.
This module detects that situation early and returns actionable fix steps.
"""
import sys
import platform


def check_environment() -> dict:
    """
    Run environment pre-flight checks.
    Returns a dict with status flags and human-readable messages.
    Call this before any SQL connection attempt.
    """
    info = {
        "python_version":    sys.version,
        "python_executable": sys.executable,
        "platform":          platform.platform(),
        "is_store_python":   "WindowsApps" in sys.executable,
    }

    try:
        import pyodbc
        info["pyodbc_status"]  = "OK"
        info["pyodbc_version"] = pyodbc.version
        drivers = pyodbc.drivers()
        info["odbc_drivers"]            = drivers
        info["sql_server_driver_found"] = any("SQL Server" in d for d in drivers)
    except ImportError:
        info["pyodbc_status"] = "NOT_INSTALLED"
        info["pyodbc_error"]  = "Run: pip install pyodbc"
    except PermissionError as e:
        info["pyodbc_status"]     = "PERMISSION_ERROR"
        info["pyodbc_error"]      = str(e)
        info["windows_alias_fix"] = True
    except OSError as e:
        info["pyodbc_status"] = "OS_ERROR"
        info["pyodbc_error"]  = str(e)
        if "Access is denied" in str(e) or "WinError 5" in str(e):
            info["windows_alias_fix"] = True
    except Exception as e:
        info["pyodbc_status"] = "ERROR"
        info["pyodbc_error"]  = str(e)

    return info


def interpret_connection_error(error: Exception) -> str:
    """
    Translate a raw ODBC/OS exception into a Markdown-formatted message
    with specific remediation steps the user can act on immediately.
    """
    err = str(error)

    if (
        "Access is denied" in err
        or isinstance(error, PermissionError)
        or "WinError 5" in err
        or "WindowsApps" in sys.executable
    ):
        return (
            "### Access Denied – Microsoft Store Python\n\n"
            "The Microsoft Store version of Python is sandboxed and cannot access "
            "the Windows registry or load ODBC driver DLLs.\n\n"
            "**Fix (one-time, ~5 minutes):**\n\n"
            "1. Open **Windows Settings → Apps → Advanced App Settings "
            "→ App Execution Aliases**\n"
            "2. Turn **OFF** both `python.exe` and `python3.exe`\n"
            "3. Download and install Python from **python.org** "
            "(check ‘Add Python to PATH’)\n"
            "4. Open a **new** terminal and run:\n"
            "   ```\n"
            "   setup_windows.bat\n"
            "   ```\n\n"
            "Until fixed, the app runs on **Demo data**."
        )

    if "IM002" in err or ("Data source" in err and "not found" in err.lower()):
        return (
            "### ODBC Driver Not Found\n\n"
            "ODBC Driver 17 for SQL Server is not installed.\n\n"
            "Download from Microsoft: https://aka.ms/downloadmsodbcsql\n\n"
            "After installing, restart the app."
        )

    if "Login failed" in err:
        return (
            "### SQL Server Login Failed\n\n"
            "Your Windows credentials were rejected.\n\n"
            "Check:\n"
            "- You are on the UPS network or connected via VPN\n"
            "- Your account has access to `IAF_Compliance`\n"
            "- Server: `SVRP000F968A.us.ups.com\\\\WWBIAP21`"
        )

    if "Network" in err or "timeout" in err.lower() or "Unable to connect" in err:
        return (
            "### Cannot Reach SQL Server\n\n"
            "The server is unreachable.\n\n"
            "Check:\n"
            "- VPN is connected\n"
            "- Server: `SVRP000F968A.us.ups.com\\\\WWBIAP21`"
        )

    return f"### Connection Error\n\n```\n{err}\n```"

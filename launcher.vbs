CreateObject("WScript.Shell").Run "pythonw.exe """ & Replace(WScript.ScriptFullName, "launcher.vbs", "app.py") & """", 0, False

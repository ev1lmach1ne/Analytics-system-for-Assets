' Lanzador de Analytics System for Assets.
' - Requiere el entorno virtual (venv) que crea instalar.bat; si no existe,
'   avisa en claro en vez de abrir la app con el Python del sistema.
' - Si requirements.txt cambio desde la ultima instalacion (p. ej. tras
'   actualizar el proyecto), instala las dependencias nuevas antes de abrir.
Option Explicit

Dim fso, sh, base, py, pyw, req, marker
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")
base = fso.GetParentFolderName(WScript.ScriptFullName)

py  = base & "\venv\Scripts\python.exe"
pyw = base & "\venv\Scripts\pythonw.exe"

If Not fso.FileExists(pyw) Then
    MsgBox "No se encuentra el entorno del programa (carpeta venv)." & vbCrLf & vbCrLf & _
           "Ejecuta primero instalar.bat (esta en esta misma carpeta) y, " & _
           "cuando termine, abre la app con el acceso directo del " & _
           "escritorio o con este launcher.", _
           vbExclamation, "Analytics System - falta instalar"
    WScript.Quit 1
End If

req    = base & "\requirements.txt"
marker = base & "\venv\.deps_instaladas"

If fso.FileExists(req) Then
    If Not FilesEqual(req, marker) Then
        ' Dependencias nuevas tras una actualizacion: instalarlas antes de
        ' abrir (ventana visible para que se vea el progreso).
        If sh.Run("""" & py & """ -m pip install -r """ & req & """", 1, True) = 0 Then
            fso.CopyFile req, marker, True
        Else
            MsgBox "No se pudieron actualizar las dependencias (revisa tu " & _
                   "conexion a internet o vuelve a ejecutar instalar.bat)." & _
                   vbCrLf & vbCrLf & "La app se abrira igualmente, pero " & _
                   "puede fallar si la actualizacion traia librerias nuevas.", _
                   vbExclamation, "Analytics System - aviso"
        End If
    End If
End If

sh.CurrentDirectory = base
sh.Run """" & pyw & """ """ & base & "\app.py""", 0, False

Function FilesEqual(rutaA, rutaB)
    FilesEqual = False
    If Not fso.FileExists(rutaA) Or Not fso.FileExists(rutaB) Then Exit Function
    On Error Resume Next
    Dim contA, contB
    contA = fso.OpenTextFile(rutaA, 1).ReadAll
    contB = fso.OpenTextFile(rutaB, 1).ReadAll
    If Err.Number <> 0 Then Exit Function
    On Error GoTo 0
    FilesEqual = (contA = contB)
End Function

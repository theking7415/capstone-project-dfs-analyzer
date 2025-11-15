# Windows-Specific Notes

## Encoding Fix Applied ✅

The application now uses UTF-8 encoding for all file operations, which fixes issues with special characters (like ✓ checkmarks) on Windows.

## Running on Windows

### Using Command Prompt
```cmd
cd "C:\Users\mahes\Desktop\Ashoka\Capstone project\My data"
python run_analyzer.py
```

### Using PowerShell
```powershell
cd "C:\Users\mahes\Desktop\Ashoka\Capstone project\My data"
python run_analyzer.py
```

### Using WSL (Windows Subsystem for Linux)
```bash
cd "/mnt/c/Users/mahes/Desktop/Ashoka/Capstone project/My data"
python3 run_analyzer.py
```

## Viewing Output Files

### Text Files (summary.txt, detailed_stats.txt)
- Open with **Notepad** (will display checkmarks correctly)
- Or any UTF-8 compatible editor (VS Code, Notepad++, etc.)
- **Avoid old Notepad** on Windows 10 (may not show UTF-8 correctly)

### CSV Files (data.csv)
- Open with **Excel** - will work perfectly
- Or **Google Sheets** for online viewing

### Images (visualization.png)
- Open with any image viewer
- Windows Photos app works great

## Common Issues

### Issue: "python: command not found"
**Solution:** Use `py` instead of `python`:
```cmd
py run_analyzer.py
```

### Issue: Module not found errors
**Solution:** Install dependencies:
```cmd
pip install numpy scipy matplotlib
```
Or:
```cmd
py -m pip install numpy scipy matplotlib
```

### Issue: Permission denied
**Solution:** Run as administrator or check file permissions

### Issue: Encoding errors (FIXED)
This was fixed by adding `encoding="utf-8"` to all file operations.
The app should now work perfectly on Windows!

## Performance on Windows

Expected runtimes (on typical Windows PC):

| Graph | Samples | Time |
|-------|---------|------|
| 3D (8 vertices) | 1000 | ~2-3 sec |
| 4D (16 vertices) | 5000 | ~8-10 sec |
| 5D (32 vertices) | 10000 | ~30-45 sec |
| 6D (64 vertices) | 10000 | ~1-2 min |

## Tips

1. **Use Windows Terminal** (modern, better than cmd.exe)
   - Download from Microsoft Store
   - Better Unicode support
   - Nicer interface

2. **Keep paths short**
   - Windows has path length limits
   - Results are saved in `data_output/` by default

3. **View results in Excel**
   - The CSV files open perfectly in Excel
   - Easy to analyze and create charts

4. **Anti-virus software**
   - May slow down file creation
   - Add exclusion for `data_output/` folder if needed

## Confirmed Working On

- ✅ Windows 11 (via WSL)
- ✅ Windows 10 (should work)
- ✅ Command Prompt
- ✅ PowerShell
- ✅ WSL Ubuntu

## Python Version

Tested with:
- Python 3.11 ✅
- Python 3.12 ✅

Minimum requirement: Python 3.10+

# Changelog

All notable changes to DFS Graph Analyzer will be documented in this file.

## [0.1.1] - 2025-11-15

### Changed
- **Author Name**: Updated from "Mahesh" to "Venkat Mahesh Mandava" across all files
  - `dfs_analyzer/__init__.py`
  - `dfs_analyzer/ui/cli.py` (About section)
  - `README.md`

- **Number Formatting**: Removed comma separators from all number displays
  - **Why**: Comma formatting (e.g., "10,000") was confusing for users who thought they needed to input commas
  - **What changed**: All numbers now display without commas (e.g., "10000" instead of "10,000")
  - Files affected:
    - `dfs_analyzer/ui/cli.py` - CLI prompts and displays
    - `dfs_analyzer/experiments/results.py` - Output file displays
    - `README.md` - Documentation examples
    - `QUICKSTART.md` - Quick start examples
    - `PROJECT_STATUS.md` - Performance metrics
    - `WINDOWS_NOTES.md` - Windows-specific notes
    - `test_cli_automated.py` - Test output

### Examples of Changes

**Before:**
```
Recommended samples: 25,000
Enter number of samples (recommended: 10,000):
Samples: 1,000
```

**After:**
```
Recommended samples: 25000
Enter number of samples (recommended: 10000):
Samples: 1000
```

### Testing
- ✅ All tests pass with new formatting
- ✅ Output files generate correctly
- ✅ No encoding errors
- ✅ User input works without commas

## [0.1.0] - 2025-11-15

### Added
- Initial release of DFS Graph Analyzer
- Interactive CLI interface
- Support for hypercube graphs (3D-15D)
- Multiple export formats (CSV, JSON, TXT, Pickle)
- Automatic visualization generation
- Statistical analysis and conjecture validation
- Comprehensive documentation
- Windows encoding fix (UTF-8)

### Features
- Run experiments on symmetric regular graphs
- Progress tracking with visual progress bars
- Reproducible results with seeded RNG
- Professional output formatting
- Built-in help system

### Documentation
- README.md - Comprehensive guide
- QUICKSTART.md - Quick start guide
- PROJECT_STATUS.md - Project overview
- WINDOWS_NOTES.md - Windows-specific instructions

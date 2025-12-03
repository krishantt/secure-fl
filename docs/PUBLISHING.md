# Publishing Guide for Secure FL

Quick reference for publishing Secure FL packages to PyPI and Docker registries.

## 🚀 Quick Start

### Check Current Version
```bash
pdm run version
# Shows: 2024.12.3.dev.1
```

### Publish to PyPI
```bash
# 1. Increment version for publishing
pdm run version-publish

# 2. Build package
pdm build

# 3. Publish to PyPI
pdm publish
```

### Publish Docker Image
```bash
# Get current version
VERSION=$(python -c "from secure_fl._version import get_version; print(get_version())")

# Build and tag
docker build -t krishantt/secure-fl:$VERSION .
docker build -t krishantt/secure-fl:latest .

# Push to registry
docker push krishantt/secure-fl:$VERSION
docker push krishantt/secure-fl:latest
```

## 📋 Version System

### Format
All versions follow: `YYYY.MM.DD.dev.INCREMENT`

Examples:
- `2024.12.3.dev.1` (first version of the day)
- `2024.12.3.dev.2` (after first publish)
- `2024.12.4.dev.1` (next day, resets to 1)

### Key Principles
- **Always `.dev`**: Every version is a development version
- **Date-based**: Easy to identify when built
- **Manual increment**: Only increments when you publish
- **Daily reset**: Each day starts with increment 1

## 🛠️ Manual Publishing (Recommended)

### PyPI Publishing Steps

1. **Check current state**:
   ```bash
   pdm run version
   pdm run version-build-info
   ```

2. **Increment for publishing**:
   ```bash
   pdm run version-publish
   ```
   This will:
   - Show current version
   - Increment the build number
   - Display new version
   - Give publishing instructions

3. **Build the package**:
   ```bash
   pdm build
   ```

4. **Test the built package**:
   ```bash
   pip install dist/*.whl
   python -c "import secure_fl; print(secure_fl.__version__)"
   ```

5. **Publish to PyPI**:
   ```bash
   pdm publish
   ```

6. **Verify on PyPI**:
   - Check https://pypi.org/project/secure-fl/
   - Test installation: `pip install secure-fl`

### Docker Publishing Steps

1. **Get version for tagging**:
   ```bash
   VERSION=$(python -c "from secure_fl._version import get_version; print(get_version())")
   echo "Publishing version: $VERSION"
   ```

2. **Build Docker image**:
   ```bash
   docker build -t krishantt/secure-fl:$VERSION .
   docker build -t krishantt/secure-fl:latest .
   ```

3. **Test the image**:
   ```bash
   docker run --rm krishantt/secure-fl:$VERSION python -c "import secure_fl; print(secure_fl.__version__)"
   ```

4. **Push to registries**:
   ```bash
   # Docker Hub
   docker push krishantt/secure-fl:$VERSION
   docker push krishantt/secure-fl:latest
   
   # GitHub Container Registry (if configured)
   docker tag krishantt/secure-fl:$VERSION ghcr.io/krishantt/secure-fl:$VERSION
   docker push ghcr.io/krishantt/secure-fl:$VERSION
   ```

## 🤖 GitHub Actions Publishing

### Manual Workflow Trigger

1. Go to **Actions** → **Manual Publishing**
2. Select options:
   - **Publish target**: `pypi`, `docker`, or `both`
   - **Increment version**: `true` (recommended)
   - **Docker tags**: Additional tags like `latest,stable`
3. Click **Run workflow**

### What the Workflow Does

- **Version Management**:
  - Increments version if requested
  - Commits the new `.build_number` file
  - Creates git tag for the version

- **PyPI Publishing**:
  - Builds package with new version
  - Tests package installation
  - Publishes to PyPI using `PYPI_API_TOKEN`
  - Creates release tag

- **Docker Publishing**:
  - Builds multi-platform images (amd64, arm64)
  - Tags with version and additional tags
  - Pushes to Docker Hub and GitHub Container Registry
  - Updates Docker Hub description

## 📁 Version Files

### Build Number File
Located at: `secure_fl/.build_number`

Format: `YYYY.MM.DD:INCREMENT`
```
2024.12.3:2
```

This file:
- Tracks the last increment used for each date
- Should be committed to git
- Only updates when publishing (not during development)

### Version Module
Located at: `secure_fl/_version.py`

Functions:
- `get_version()` - Current version string
- `get_build_info()` - Detailed build information
- `increment_build_number()` - Increment for publishing
- `print_version_info()` - Display formatted info

## 🔧 Common Commands

### Version Management
```bash
# Show current version
pdm run version

# Show build file info
pdm run version-build-info

# Increment for publishing
pdm run version-increment

# Set specific increment
pdm run version-set 5

# Prepare for publishing (increment + instructions)
pdm run version-publish
```

### Package Operations
```bash
# Build package
pdm build

# Test built package
pip install dist/*.whl

# Check version consistency
python -c "
import secure_fl
from secure_fl._version import get_version
print(f'Package: {secure_fl.__version__}')
print(f'Module:  {get_version()}')
print(f'Match:   {secure_fl.__version__ == get_version()}')
"
```

### Development Workflow
```bash
# Start of day - check version
pdm run version  # e.g., 2024.12.3.dev.1

# Develop all day (version stays same)
# ... make changes, test, commit ...

# Ready to publish first version
pdm run version-publish  # increments to 2024.12.3.dev.2
pdm build
pdm publish

# Continue development
# ... more changes ...

# Ready for second publish
pdm run version-publish  # increments to 2024.12.3.dev.3
pdm build
pdm publish
```

## 🚨 Important Notes

### Do NOT
- ❌ Manually edit `.build_number` file
- ❌ Increment version unless actually publishing
- ❌ Use automated publishing for production
- ❌ Publish without testing the built package

### Always DO
- ✅ Test package before publishing: `pip install dist/*.whl`
- ✅ Commit `.build_number` file changes
- ✅ Use `version-publish` script for increments
- ✅ Verify version on PyPI after publishing
- ✅ Document significant changes in commit messages

### Environment Setup

Required secrets for GitHub Actions:
- `PYPI_API_TOKEN`: PyPI API token for publishing
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password/token

Required tools for local publishing:
- PDM: `pip install pdm`
- Docker: For container publishing
- PyPI credentials: Configure with `pdm config`

## 🔍 Troubleshooting

### Version Not Incrementing
```bash
# Check build file
pdm run version-build-info

# Reset if needed (caution: this resets to increment 1)
rm secure_fl/.build_number
```

### Package Version Mismatch
```bash
# Rebuild after version changes
pdm build

# Clear PDM cache
pdm cache clear

# Reinstall in development mode
pip install -e .
```

### Publishing Fails
```bash
# Check credentials
pdm config

# Test build first
pdm build
twine check dist/*

# Verify package installs
pip install dist/*.whl
```

### Docker Issues
```bash
# Test local build
docker build -t test-secure-fl .
docker run --rm test-secure-fl python -c "import secure_fl; print(secure_fl.__version__)"

# Check registry credentials
docker login
```

## 📚 References

- [Version System Documentation](docs/VERSION_SYSTEM.md)
- [PyPI Package](https://pypi.org/project/secure-fl/)
- [Docker Hub](https://hub.docker.com/r/krishantt/secure-fl)
- [GitHub Repository](https://github.com/krishantt/secure-fl)
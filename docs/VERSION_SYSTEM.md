# Version System Documentation

This document describes the simple development versioning system used in the Secure FL project.

## Overview

Secure FL uses a development-focused versioning system that generates version numbers based on the current date and a build increment. This is designed for continuous development with manual publishing control.

## Version Format

The version follows this simple pattern:
```
YYYY.MM.DD.dev.INCREMENT
```

Where:
- `YYYY.MM.DD`: Current date (e.g., `2024.12.03`)
- `dev`: Always present - indicates development version
- `INCREMENT`: Sequential number for the day, incremented only when publishing

### Examples

- **Development**: `2024.12.3.dev.1`
- **After Publishing**: `2024.12.3.dev.2`
- **Next Day**: `2024.12.4.dev.1`

## Development Philosophy

This system is designed with these principles:

- **Always Dev**: Every version is a development version unless manually published
- **Date-Based**: Easy to see when code was built
- **Manual Publishing**: Increment only happens when you decide to publish
- **Simple**: No complex CI detection or environment switching

## Usage

### Command Line Interface

The project includes a version management script at `scripts/version.py`:

```bash
# Show current version information
python scripts/version.py show

# Increment build increment (for publishing only)
python scripts/version.py increment

# Set specific build increment
python scripts/version.py set 42

# Show build file information
python scripts/version.py build-info

# Prepare for publishing (increments version)
python scripts/version.py publish
```

### PDM Scripts

For convenience, PDM scripts are available:

```bash
# Show version
pdm run version

# Increment version for publishing
pdm run version-increment

# Set specific build increment
pdm run version-set 42

# Show build file info
pdm run version-build-info

# Prepare for publishing
pdm run version-publish
```

### Python API

```python
from secure_fl._version import get_version, get_build_info, print_version_info

# Get version string
version = get_version()
print(f"Version: {version}")

# Get detailed build information
info = get_build_info()
print(f"Build Date: {info['build_date']}")
print(f"Is CI: {info['is_ci']}")

# Print formatted version info
print_version_info()
```

### Package Import

```python
import secure_fl

# Version is available as package attribute
print(f"Secure FL Version: {secure_fl.__version__}")
# Output: Secure FL Version: 2024.12.3.dev.1
```

## Publishing Workflow

### Manual Publishing

Publishing is intentionally manual to maintain control:

1. **Prepare for Publishing**:
   ```bash
   pdm run version-publish  # Increments version
   ```

2. **Build Package**:
   ```bash
   pdm build
   ```

3. **Publish to PyPI**:
   ```bash
   pdm publish
   ```

4. **Docker (when needed)**:
   ```bash
   docker build -t krishantt/secure-fl:$(python -c "from secure_fl._version import get_version; print(get_version())") .
   docker push krishantt/secure-fl:latest
   ```

### GitHub Actions Publishing

Use the manual publishing workflow:

1. Go to Actions → Manual Publishing
2. Choose what to publish (PyPI, Docker, or both)
3. Select whether to increment version
4. Run the workflow

This will:
- Increment version if requested
- Build and publish to selected targets
- Create git tags
- Update package repositories

## Build System Integration

### pyproject.toml Configuration

The version is dynamically read from the `_version.py` module:

```toml
[project]
dynamic = ["version"]

[tool.pdm]
version = {source = "call", getter = "secure_fl._version:get_version"}
```

### Package Building

When building packages with PDM:

```bash
pdm build
```

The built package will contain the current dynamic version, ensuring traceability.

## Version Storage

### Build Increment Persistence

Build increments are stored in `secure_fl/.build_number` with format:
```
YYYY.MM.DD:INCREMENT
```

This file tracks the highest increment used for each date and is only updated when publishing.

### Version History

Each published version can be traced back to:
- Exact date of publishing
- Build increment for that date
- Git commit and tag (created during publishing)

## Development Workflow

### Daily Development

1. **Check current version**: `pdm run version`
   - Same version all day until you publish
2. **Build and test**: `pdm build && pip install dist/*.whl`
3. **Develop and iterate**: Version stays the same

### Publishing Process

1. **Ready to publish**: `pdm run version-publish`
   - This increments the build number
2. **Build final package**: `pdm build`
3. **Publish to PyPI**: `pdm publish`
   - Or use GitHub Actions manual workflow

### Version Management

- **Never increment manually** unless actually publishing
- **Build increment only changes** when you publish
- **New day = new base version** (increment resets to 1)

## Troubleshooting

### Common Issues

**Version not incrementing:**
- Remember: increment only happens when publishing
- Use `pdm run version-publish` to increment
- Check `.build_number` file exists and is readable

**Build increment resets to 1:**
- This is normal for a new date
- Each day starts with increment 1

**Package version mismatch:**
- Rebuild package after version changes: `pdm build`
- Clear PDM cache: `pdm cache clear`
- Verify PDM configuration in `pyproject.toml`

### Debugging Commands

```bash
# Show build file information
pdm run version-build-info

# Test version consistency
python -c "
import secure_fl
from secure_fl._version import get_version
print(f'Package: {secure_fl.__version__}')
print(f'Module:  {get_version()}')
print(f'Match:   {secure_fl.__version__ == get_version()}')
"

# Check build file directly
cat secure_fl/.build_number

# Show current version details
pdm run version
```

## Best Practices

### For Developers

- **Check version regularly**: `pdm run version`
- **Only increment when publishing**: Don't increment for testing
- **Never edit `.build_number` manually**: Use the scripts
- **Build after any version change**: `pdm build`

### For Publishing

- **Use version-publish script**: `pdm run version-publish`
- **Publish same day as increment**: Don't let incremented versions sit
- **Use manual workflows**: Avoid automatic publishing
- **Test before publishing**: Always test the exact package you'll publish

### For Team Collaboration

- **Commit `.build_number` file**: It tracks published versions
- **Coordinate publishing**: Only one person should publish per day
- **Document releases**: Note what changed with each increment
- **Use descriptive commit messages**: When incrementing versions

## Security Considerations

The version system is designed to be safe:
- No sensitive information in version strings
- Build increments are just sequential numbers
- All version data is safe for public repositories
- No external dependencies for version generation

## Example Workflow

Here's a typical development and publishing cycle:

```bash
# Day 1: Start development
pdm run version        # Shows: 2024.12.3.dev.1
# ... develop all day ...
pdm run version        # Still: 2024.12.3.dev.1

# Ready to publish first version of the day
pdm run version-publish # Increments to: 2024.12.3.dev.1 → 2024.12.3.dev.2
pdm build
pdm publish

# Continue development
# ... more changes ...

# Ready for another publish
pdm run version-publish # Increments to: 2024.12.3.dev.3
pdm build
pdm publish

# Day 2: New date
pdm run version        # Shows: 2024.12.4.dev.1 (reset for new day)
```

This keeps versions simple, predictable, and tied to actual publishing events.
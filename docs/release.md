# Release

Releases are triggered by a **Version Upgrade PR** that bumps the version in `pyproject.toml`. Merging to `main` automatically creates the git tag and publishes the wheel to [GitHub Releases](https://github.com/openecos-projects/ecc-dreamplace/releases).

## Steps

**1. Bump version** in `pyproject.toml`:

```toml
[project]
name = "ecc-dreamplace"
version = "0.2.0"  # bump this line
```

Also update `MODULE.bazel` to match — the CI version consistency check will fail otherwise.

[PEP 440](https://peps.python.org/pep-0440/) formats are supported:
- `0.1.0`, `0.2.0`, `1.0.0` - stable releases
- `0.1.0a1` / `0.1.0-alpha.1` - alpha
- `0.1.0b1` / `0.1.0-beta.1` - beta
- `0.1.0rc1` / `0.1.0-rc.1` - release candidate

**2. Create and merge PR**:

```bash
git checkout -b bump-version-0.2.0
git commit -am "chore: bump version to 0.2.0"
gh pr create --title "chore: bump version to 0.2.0" --body "Preparing release v0.2.0"
# Wait for CI to pass, then merge to main
```

**3. Automatic (no action needed)**:

After merge to `main`:
1. **Auto-tag**: `auto-tag.yml` detects the version change and creates `v0.2.0` (skipped if tag exists).
2. **Release**: `release.yml` triggers on the `v*` tag, builds the wheel, generates release notes with `git-cliff` using `.github/cliff.toml`, and publishes to GitHub Releases.

## Verify

```bash
gh release list
```

## Rollback

```bash
gh release delete v0.2.0 --cleanup-tag  # removes release and git tag
git push origin :refs/tags/v0.2.0        # force-delete tag if still present
```

Then merge a revert PR to restore the previous version.

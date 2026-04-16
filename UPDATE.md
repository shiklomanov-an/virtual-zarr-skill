# Skill Update Log

## 2026-04-16: Icechunk 2.0 + VirtualiZarr 2.5

Updated the virtual-zarr skill to reflect breaking changes and new features in:
- **icechunk 2.0.3** (released 2026-04-16)
- **VirtualiZarr 2.5.1** (released 2026-04-09)

### Library Versions Referenced

| Package | Version | Release Date |
|---------|---------|--------------|
| icechunk | 2.0.3 | 2026-04-16 |
| virtualizarr | 2.5.1 | 2026-04-09 |

### Changes Made

#### SKIll.md

1. **Quick Start write pattern**: Updated to use explicit session pattern (`session = repo.writable_session(); vds.vz.to_icechunk(session.store); session.commit(...)`) as the recommended approach, with `repo.transaction()` shown as a valid alternative.

2. **Workflow 1 & 2**: Updated icechunk write calls to use explicit session pattern.

3. **Workflow 4 (Append)**: Updated to use explicit session pattern.

4. **Workflow 5 (Reading)**: Updated `ManifestStore` usage to omit `consolidated=False` (no longer needed since VirtualiZarr 2.3).

5. **ZarrParser description**: Added note about async listing performance improvement.

6. **New `to_icechunk()` signature**: Added `region=` parameter to API reference.

7. **Icechunk API reference**: Updated `session.commit()` signature to show `allow_empty=False`; added `in_memory_storage()` and `upgrade_icechunk_repository()`.

8. **New features section**: Added "New Features (VirtualiZarr 2.x / Icechunk 2.x)" section covering:
   - Region writes (`to_icechunk(store, region=...)`)
   - Sharding support
   - DataTree support (`open_virtual_datatree()`)
   - ZarrParser performance
   - `in_memory_storage()` for testing
   - Redesigned ForkSession for distributed writes

#### examples.md

1. **Python version requirement**: Updated from "3.10 or higher" to "3.12 or higher" (icechunk 2.0 requirement).

2. **Example 7 (Distributed Parallel Writes)**: Updated ForkSession pattern:
   - Now creates one `session.fork()` per worker (instead of sharing one fork's bytes across all workers)
   - Merge uses `session.merge(fork)` called sequentially per fork (instead of `session.merge(*forks)`)
   - Worker function unchanged (pickle round-trip still works)

#### troubleshooting.md

No changes required. All `repo.transaction()` patterns remain valid in icechunk 2.0.

### Icechunk 2.0 Breaking Changes Summary

For reference, key breaking changes in icechunk 2.0 (from 1.x):

1. **Python 3.12+ required** (dropped 3.11 support)
2. **`ChunkType` enum is now `snake_case`**: `ChunkType.virtual`, `ChunkType.inline`, `ChunkType.native`, `ChunkType.uninitialized` (was UPPER_CASE in 1.x)
3. **`Repository.manifest_files()` renamed to `Repository.list_manifest_files()`**
4. **`Session.commit()` keyword-only args**: `rebase_with`, `rebase_tries` are now keyword-only; `allow_empty=False` added
5. **`Session.flush()` keyword-only args**: `metadata` is now keyword-only
6. **`inspect_snapshot()` returns `dict`** (was JSON string; `pretty` parameter removed)
7. **`ForkSession` redesigned**: Use `session.fork()` → picklable ForkSession; merge with `Session.merge(fork_session)` (not `ForkSession.merge(Session)`); no commit needed before forking
8. **`list_objects` deprecated** in favor of `list_objects_metadata`
9. **New `in_memory_storage()` helper** for ephemeral repos
10. **New `upgrade_icechunk_repository(repo, dry_run=False)`** to migrate v1 repos

### VirtualiZarr 2.x Highlights

Key additions since the skill was originally written:

- **2.5.1**: Sharding support, Azure URL fixes
- **2.5.0**: `to_icechunk(region=...)` parameter, `ZarrParser` async listing performance
- **2.4.0**: `ObjectStoreRegistry` moved to `obspec_utils` package
- **2.3.0**: `open_virtual_datatree()` top-level function; `ManifestStore` no longer requires `consolidated=False`
- **2.2.0**: `ZarrParser` added; Zarr V2 parser; virtual TIFF integration (`virtual-tiff`)
- **2.1.0**: `to_icechunk()` now raises on missing virtual chunk containers (was silent drop)

### How to Update the Skill

> **Important**: All edits should be made to the skill files in **this project directory** (`virtual-zarr/`), not in the default skill storage location (`~/.agents/skills/virtual-zarr/`). The system-level skill installation is managed separately and will be updated via a dedicated mechanism.

1. Review release notes for new library versions:
   - https://github.com/earth-mover/icechunk/releases
   - https://github.com/zarr-developers/VirtualiZarr/releases
2. Check for breaking changes in API signatures and patterns
3. Update `SKIll.md` with corrected patterns and new features
4. Update `examples.md` with corrected runnable code
5. Update `troubleshooting.md` if error messages or solutions change
6. Document changes in this file

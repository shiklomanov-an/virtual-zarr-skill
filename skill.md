# Virtual Zarr Agent Skill

AI agent skill for creating and working with virtual Zarr stores using VirtualiZarr and icechunk.

## Overview

This skill enables AI agents to help users virtualize legacy archival data formats (NetCDF, HDF5, GRIB, FITS) into cloud-optimized Zarr stores without copying the underlying data. It covers:

- **VirtualiZarr**: Creates "virtual" Zarr stores by generating lightweight references (manifests) to chunks in existing files
- **Icechunk**: Provides transactional storage with ACID guarantees, version control (branches, tags, snapshots), and time-travel capabilities for Zarr data

### When to Use This Skill

Activate this skill when users need to:
- Convert NetCDF/HDF5 files to virtual Zarr references
- Aggregate multiple archival files into unified virtual datasets
- Set up version-controlled array storage with git-like operations
- Access legacy data in cloud-optimized ways without data duplication
- Create kerchunk references (legacy format)

### Key Benefits

- **No data copying**: Virtual references point to existing files
- **Fast aggregation**: Combine thousands of files in seconds
- **Version control**: Track changes to array data over time
- **Cloud-optimized**: Efficient parallel access to chunks
- **Storage agnostic**: Works with S3, GCS, Azure, local filesystems

## Core Concepts

### Virtual vs. Loadable Variables

When virtualizing data, variables can be:

- **Virtual**: Large arrays stored as references to chunks in original files (ManifestArray objects). The actual data stays in the source files.
- **Loadable**: Small arrays loaded into memory (NumPy arrays). Typically used for coordinates, metadata, or variables that need special handling.

By default, dimension coordinates are loaded into memory, while data variables remain virtual.

### Chunk Manifests

A `ChunkManifest` is a mapping of chunk keys to their locations:

```python
{
    "0.0.0": {"path": "s3://bucket/file.nc", "offset": 1024, "length": 8192},
    "0.0.1": {"path": "s3://bucket/file.nc", "offset": 9216, "length": 8192},
}
```

These manifests enable Zarr libraries to fetch specific chunks from the original files on-demand.

### Object Store Registry

An `ObjectStoreRegistry` maps URL prefixes to `ObjectStore` instances that handle data access:

```python
from obstore.store import S3Store
from obspec_utils.registry import ObjectStoreRegistry

bucket = "s3://my-bucket"
store = S3Store(bucket="my-bucket", region="us-east-1")
registry = ObjectStoreRegistry({bucket: store})
```

The registry routes requests to the appropriate storage backend based on URL prefixes.

### Parsers

Parsers convert file formats to virtual Zarr representations. Available parsers:

- `HDFParser`: NetCDF4/HDF5 files (most common)
- `NetCDF3Parser`: NetCDF3 files
- `FITSParser`: FITS astronomical data
- `DMRPPParser`: DMR++ sidecar files
- `ZarrParser`: Existing Zarr stores
- `KerchunkJSONParser` / `KerchunkParquetParser`: Existing kerchunk references

### Homogeneity Requirements

Data can only be virtualized if files have:

1. **Homogeneous chunk shapes**: All files must use identical chunking
2. **Homogeneous codecs**: Same compression/encoding across files
3. **Homogeneous dtypes**: Consistent data types
4. **Supported codecs**: Compression must be recognized by Zarr

See [troubleshooting.md](troubleshooting.md) for detailed error handling when these requirements aren't met.

## Quick Start

Minimal example virtualizing a single NetCDF file from S3 and writing to icechunk:

```python
import icechunk
import virtualizarr as vz
from obstore.store import from_url
from obspec_utils.registry import ObjectStoreRegistry
from virtualizarr.parsers import HDFParser

# Setup object store for source data (public bucket)
bucket = "s3://noaa-goes16"
source_store = from_url(bucket, region="us-east-1", skip_signature=True)
registry = ObjectStoreRegistry({bucket: source_store})

# Virtualize a single file
url = f"{bucket}/ABI-L2-MCMIPF/2024/099/18/OR_ABI-L2-MCMIPF-M6_G16_s20240991800204_e20240991809524_c20240991810005.nc"
parser = HDFParser()
vds = vz.open_virtual_dataset(
    url,
    registry=registry,
    parser=parser,
    loadable_variables=['time', 'x', 'y'],  # Load coordinates
)

# Write to icechunk repository
storage = icechunk.local_filesystem_storage("/tmp/my-repo")
repo = icechunk.Repository.create(storage)

with repo.transaction("main", message="Initial virtualization") as store:
    vds.vz.to_icechunk(store)
```

## Common Workflows

### Workflow 1: Single File Virtualization

Virtualize one file and write to icechunk:

```python
import icechunk
import virtualizarr as vz
from obstore.store import S3Store
from obspec_utils.registry import ObjectStoreRegistry
from virtualizarr.parsers import HDFParser

# Setup source storage
bucket = "s3://my-data-bucket"
store = S3Store(bucket="my-data-bucket", region="us-west-2")
registry = ObjectStoreRegistry({bucket: store})

# Virtualize
url = f"{bucket}/climate/temperature_2024.nc"
parser = HDFParser(drop_variables=['bounds', 'crs'])  # Drop unwanted variables
vds = vz.open_virtual_dataset(
    url,
    registry=registry,
    parser=parser,
    loadable_variables=None,  # Default: load dimension coordinates only
    decode_times=True,  # Decode CF time variables (requires loading them)
)

# Setup icechunk repository
storage = icechunk.s3_storage(
    bucket="my-icechunk-repo",
    prefix="datasets/temperature",
    region="us-west-2",
    from_env=True,  # Use IAM role credentials
)

# Configure virtual chunk access
config = icechunk.RepositoryConfig.default()
virtual_store_config = icechunk.s3_store(region="us-west-2")
container = icechunk.VirtualChunkContainer(bucket, virtual_store_config)
config.set_virtual_chunk_container(container)

# Authorize virtual chunk access
credentials = icechunk.containers_credentials({
    bucket: icechunk.s3_credentials(from_env=True)
})

# Create repository and write
repo = icechunk.Repository.create(storage, config, credentials)
with repo.transaction("main", message="Add temperature data") as store:
    vds.vz.to_icechunk(store)
```

### Workflow 2: Multi-File Virtualization

Aggregate multiple files along a dimension:

```python
import virtualizarr as vz
from obstore.store import from_url
from obspec_utils.registry import ObjectStoreRegistry
from virtualizarr.parsers import HDFParser

# Setup
bucket = "s3://climate-archive"
store = from_url(bucket, region="us-east-1")
registry = ObjectStoreRegistry({bucket: store})
parser = HDFParser()

# List of files to aggregate (time series)
urls = [
    f"{bucket}/daily/temp_2024_001.nc",
    f"{bucket}/daily/temp_2024_002.nc",
    f"{bucket}/daily/temp_2024_003.nc",
    # ... more files
]

# Method 1: Using open_virtual_mfdataset
combined = vz.open_virtual_mfdataset(
    urls,
    registry=registry,
    parser=parser,
    combine='by_coords',  # Automatically order by coordinate values
    concat_dim='time',
    parallel=False,  # Set to 'dask' or ThreadPoolExecutor for parallel
    loadable_variables=['time', 'lat', 'lon'],
    decode_times=True,
)

# Method 2: Manual concatenation with xarray
import xarray as xr

datasets = []
for url in urls:
    vds = vz.open_virtual_dataset(url, registry=registry, parser=parser)
    datasets.append(vds)

combined = xr.concat(
    datasets,
    dim='time',
    coords='minimal',  # Don't broadcast coordinates
    compat='override',  # Allow minor inconsistencies
)

# Write combined dataset
with repo.transaction("main", message="Aggregated daily data") as store:
    combined.vz.to_icechunk(store)
```

### Workflow 3: Icechunk Version Control

Use git-like version control operations:

```python
import icechunk
import zarr

# Open existing repository
storage = icechunk.local_filesystem_storage("/data/my-repo")
repo = icechunk.Repository.open(storage)

# Create a new branch for experimental changes
session = repo.writable_session("main")
session.commit("Base version")
repo.create_branch("experiment", session.snapshot_id)

# Make changes on the branch
session = repo.writable_session("experiment")
group = zarr.open_group(session.store, mode="r+")
group["temperature"][:10, :10] = 0  # Modify data
session.commit("Zero out test region")

# View history
for snapshot in repo.ancestry("experiment"):
    print(f"{snapshot.id}: {snapshot.message} at {snapshot.written_at}")

# Tag an important version
repo.create_tag("v1.0.0", snapshot_id=session.snapshot_id)

# Time travel: read from specific version
old_session = repo.readonly_session(tag="v1.0.0")
old_data = zarr.open_array(old_session.store, path="temperature", mode="r")[:]

# Compare versions
diff = repo.diff(
    from_snapshot_id=old_session.snapshot_id,
    to_snapshot_id=session.snapshot_id
)
```

### Workflow 4: Appending Data to Icechunk

Incrementally add data along a dimension:

```python
import icechunk
import virtualizarr as vz

# Initial dataset
vds1 = vz.open_virtual_dataset(
    "s3://archive/month_01.nc",
    registry=registry,
    parser=parser
)

storage = icechunk.local_filesystem_storage("/data/timeseries-repo")
repo = icechunk.Repository.create(storage)

# Write initial data
with repo.transaction("main", message="January data") as store:
    vds1.vz.to_icechunk(store)

# Append February data
vds2 = vz.open_virtual_dataset(
    "s3://archive/month_02.nc",
    registry=registry,
    parser=parser
)

with repo.transaction("main", message="Added February") as store:
    vds2.vz.to_icechunk(store, append_dim="time")

# Continue appending more months...
```

### Workflow 5: Reading Virtual Datasets

Multiple ways to read virtualized data:

```python
import xarray as xr
import zarr

# Method 1: Read via icechunk session (recommended)
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store, consolidated=False, zarr_format=3)
data = ds["temperature"].values  # Triggers actual data fetch

# Method 2: Read via ManifestStore (without writing to icechunk)
from virtualizarr.manifests.store import ManifestStore

parser = HDFParser()
manifest_store = parser(url, registry)
ds = xr.open_zarr(manifest_store, consolidated=False, zarr_format=3)

# Method 3: Load specific chunks using Zarr directly
session = repo.readonly_session("main")
array = zarr.open_array(session.store, path="temperature", mode="r")
chunk_data = array.get_basic_selection((slice(0, 10), slice(0, 10)))
```

## Storage Backends

### S3 Storage

Three authentication patterns for S3 access:

#### Public Buckets (Anonymous)

For public datasets, skip credential signing:

```python
from obstore.store import from_url

# VirtualiZarr: skip_signature=True
store = from_url("s3://noaa-goes16", region="us-east-1", skip_signature=True)

# Icechunk: anonymous=True
storage = icechunk.s3_storage(
    bucket="my-repo",
    prefix="data",
    region="us-east-1",
    anonymous=True,
)

# For virtual chunks
virtual_store = icechunk.s3_store(region="us-east-1", anonymous=True)
```

#### Instance Profile / IAM Role

For EC2, ECS, Lambda with IAM roles attached:

```python
# VirtualiZarr: default behavior (omit credentials)
store = from_url("s3://my-bucket", region="us-west-2")

# Icechunk: from_env=True or omit credentials
storage = icechunk.s3_storage(
    bucket="my-repo",
    prefix="data",
    region="us-west-2",
    from_env=True,  # Explicit
)

# Or let it default (same effect)
storage = icechunk.s3_storage(
    bucket="my-repo",
    prefix="data",
    region="us-west-2",
)
```

#### Explicit Credentials

Using AWS access keys (store in environment variables):

```python
import os
from obstore.store import S3Store

# VirtualiZarr
store = S3Store.from_url(
    "s3://my-bucket",
    access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)

# Icechunk
storage = icechunk.s3_storage(
    bucket="my-repo",
    prefix="data",
    region="us-west-2",
    access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    session_token=os.environ.get("AWS_SESSION_TOKEN"),  # Optional
)
```

**Security Best Practice**: Never hardcode credentials. Always use environment variables or IAM roles.

### Local Filesystem

For local files:

```python
from obstore.store import LocalStore
from pathlib import Path

# VirtualiZarr
store_path = Path("/data/archive")
file_url = f"file://{store_path}/temperature.nc"
store = LocalStore(prefix=store_path)
registry = ObjectStoreRegistry({f"file://{store_path}": store})

# Icechunk
storage = icechunk.local_filesystem_storage("/data/icechunk-repo")
```

**Note**: Local filesystem storage in icechunk is not safe for concurrent writes. Use S3/GCS/Azure for production.

### Other Backends

**GCS (Google Cloud Storage)**:
```python
# Similar patterns with gcs_storage() and from_env=True
storage = icechunk.gcs_storage(
    bucket="my-bucket",
    prefix="data",
    from_env=True,  # or service_account_file=, bearer_token=, etc.
)
```

**Azure Blob Storage**:
```python
storage = icechunk.azure_storage(
    account="myaccount",
    container="mycontainer",
    prefix="data",
    from_env=True,  # or access_key=, sas_token=, etc.
)
```

See icechunk documentation for full details on GCS and Azure authentication options.

## Key Parameters and Options

### loadable_variables

Controls which variables are loaded into memory vs. kept virtual:

- `None` (default): Load dimension coordinates only
- `[]`: Nothing loadable (fully virtual)
- `['var1', 'var2']`: Load specific variables

**When to load variables**:
- Small arrays (coordinates, scalar metadata)
- Variables needing decoding (times, categoricals)
- Variables with inconsistent chunking across files
- Coordinates needed for `combine='by_coords'`

### decode_times

Whether to decode CF time variables:

- `None` (default): No decoding
- `True`: Decode time variables (requires them to be loadable)

```python
vds = vz.open_virtual_dataset(
    url,
    registry=registry,
    parser=parser,
    loadable_variables=['time'],  # Must load to decode
    decode_times=True,
)
```

### combine modes (open_virtual_mfdataset)

- `'by_coords'`: Automatically order datasets by coordinate values (requires loadable coordinates)
- `'nested'`: Manual ordering via nested list structure

```python
# by_coords: automatic ordering
combined = vz.open_virtual_mfdataset(
    urls,
    registry=registry,
    parser=parser,
    combine='by_coords',
    loadable_variables=['time'],  # Needed for by_coords
)

# nested: manual control
combined = vz.open_virtual_mfdataset(
    [[url1, url2], [url3, url4]],  # 2D structure
    registry=registry,
    parser=parser,
    combine='nested',
    concat_dim=['ensemble', 'time'],
)
```

### parallel execution

Options for parallel file processing in `open_virtual_mfdataset`:

- `False` (default): Serial execution
- `'dask'`: Use dask.delayed
- `ThreadPoolExecutor()` / `ProcessPoolExecutor()`: Thread/process pools

```python
from concurrent.futures import ThreadPoolExecutor

combined = vz.open_virtual_mfdataset(
    urls,
    registry=registry,
    parser=parser,
    combine='by_coords',
    parallel=ThreadPoolExecutor(max_workers=10),
)
```

### Virtual Chunk Containers (Icechunk)

Configure access to original data files:

```python
config = icechunk.RepositoryConfig.default()

# Add virtual chunk container
virtual_store = icechunk.s3_store(region="us-west-2", anonymous=True)
container = icechunk.VirtualChunkContainer(
    "s3://source-bucket/",  # URL prefix for virtual chunks
    virtual_store
)
config.set_virtual_chunk_container(container)

# Authorize access
credentials = icechunk.containers_credentials({
    "s3://source-bucket/": icechunk.s3_credentials(anonymous=True)
})

repo = icechunk.Repository.create(storage, config, credentials)
```

## Validation and Requirements

Before virtualizing, ensure data meets these requirements:

**Quick Validation Checklist**:

1. **Chunk shape consistency**: All files use same chunk dimensions
2. **Codec compatibility**: Same compression/encoding (gzip, zlib, etc.)
3. **Data type consistency**: Same dtype across files
4. **Supported codecs**: Check Zarr recognizes the compression

**Quick validation command**:

```python
import h5py

# Check chunking and compression
with h5py.File("file.nc", "r") as f:
    var = f["temperature"]
    print(f"Chunks: {var.chunks}")
    print(f"Compression: {var.compression}")
    print(f"Dtype: {var.dtype}")
```

**Common issues**:
- Mixed chunk sizes → Rechunk files or filter to homogeneous subset
- Unsupported compression → Decompress and recompress with supported codec
- Mixed dtypes → Convert to common dtype or filter files

See [troubleshooting.md](troubleshooting.md) for detailed error diagnosis and solutions.

## Best Practices

### Performance Optimization

1. **Use parallel processing** for large file collections:
   ```python
   combined = vz.open_virtual_mfdataset(urls, parallel='dask', ...)
   ```

2. **Load only necessary variables**: Keep small metadata loaded, large data virtual

3. **Use caching stores** for repeated access:
   ```python
   from obspec_utils.cache import CachingReadableStore
   cached_store = CachingReadableStore(base_store, max_size=1_000_000_000)
   ```

4. **Batch operations** when processing many files to manage memory

### When to Load vs. Virtualize

**Load into memory**:
- Dimension coordinates (< 1MB typically)
- Scalar metadata variables
- Variables needing transformation (time decoding)
- Frequently accessed small arrays

**Keep virtual**:
- Large data variables (> 100MB)
- Infrequently accessed arrays
- Data with consistent chunking
- Raw measurement data

### Icechunk vs. Kerchunk

**Use Icechunk (recommended)**:
- Need version control or time-travel
- Multiple writers/concurrent access
- Production workflows
- Cloud-native deployment

**Use Kerchunk (legacy)**:
- Existing kerchunk-based workflows
- Simple read-only references
- No versioning needed

```python
# Kerchunk output (legacy, less common)
vds.vz.to_kerchunk("references.json", format="json")
```

### Chunking Strategy

For optimal performance:
- Chunk size: 10-100 MB per chunk (balance parallelism and overhead)
- Align chunks with access patterns (e.g., time slices vs. spatial tiles)
- Match chunk dimensions to analysis needs

## Reference

### Complete Examples

See [examples.md](examples.md) for full runnable code examples covering:
- Public S3 datasets
- Private S3 with instance profile
- Private S3 with explicit credentials
- Multi-file time series aggregation
- Icechunk version control workflows
- Distributed parallel writes

### Troubleshooting

See [troubleshooting.md](troubleshooting.md) for detailed help with:
- Data homogeneity errors
- Storage and authentication issues
- Parser errors
- Icechunk conflicts
- Performance problems

### Quick API Reference

**VirtualiZarr Main Functions**:
- `open_virtual_dataset(url, registry, parser, ...)` → Single file
- `open_virtual_mfdataset(urls, registry, parser, ...)` → Multiple files
- `open_virtual_datatree(url, registry, parser, ...)` → Hierarchical data

**Xarray Accessors**:
- `ds.vz.to_icechunk(store, append_dim=None)` → Write to icechunk
- `ds.vz.to_kerchunk(filepath, format='json')` → Write kerchunk refs
- `ds.vz.rename_paths(new)` → Update file paths
- `ds.vz.nbytes` → Virtual reference size

**Icechunk Main Classes**:
- `Repository.create(storage)` / `.open(storage)` → Repo management
- `repo.writable_session(branch)` → Create writable session
- `repo.readonly_session(branch/tag/snapshot_id)` → Read-only access
- `repo.transaction(branch, message)` → Context manager for commits
- `session.commit(message)` → Create snapshot
- `repo.create_branch(name, snapshot_id)` / `.create_tag(name, snapshot_id)`

### External Documentation

- [VirtualiZarr Documentation](https://virtualizarr.readthedocs.io/)
- [Icechunk Documentation](https://icechunk.io/)
- [Zarr Specification](https://zarr-specs.readthedocs.io/)
- [Xarray Documentation](https://docs.xarray.dev/)

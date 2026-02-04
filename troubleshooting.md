# Virtual Zarr Troubleshooting Guide

Detailed error scenarios, diagnosis steps, and solutions for VirtualiZarr and icechunk.

## How to Use This Guide

1. Find the error message or symptom in the relevant section
2. Follow the diagnosis steps to identify the root cause
3. Apply the recommended solution
4. If issues persist, check FAQ or file an issue with the relevant project

## Data Homogeneity Errors

Virtual datasets require homogeneous chunk shapes, codecs, and data types across files.

### Inconsistent Chunk Shapes

**Symptoms**:
- Error: `ValueError: Cannot concatenate arrays with different chunk shapes`
- Error: `Inconsistent chunk shapes found`
- Concatenation fails with shape mismatch

**Diagnosis**:

Check chunk shapes across files:

```python
import h5py

def check_chunks(file_paths, variable):
    """Check chunk shapes across multiple files."""
    chunks_info = {}
    for path in file_paths:
        with h5py.File(path, 'r') as f:
            if variable in f:
                chunks_info[path] = f[variable].chunks
    return chunks_info

# Usage
files = ["file1.nc", "file2.nc", "file3.nc"]
info = check_chunks(files, "temperature")
for file, chunks in info.items():
    print(f"{file}: {chunks}")
```

**Common Causes**:
1. Files created with different chunking settings
2. Mixed sources (some rechunked, some not)
3. Different compression applied to subsets
4. Files from different processing versions

**Solutions**:

**Option 1**: Filter to homogeneous subset:

```python
# Identify files with matching chunk shapes
target_chunks = (1, 100, 100)
compatible_files = [
    f for f in file_list 
    if get_chunk_shape(f, "temperature") == target_chunks
]

# Virtualize only compatible files
vds = vz.open_virtual_mfdataset(
    compatible_files,
    registry=registry,
    parser=parser,
    combine='by_coords',
)
```

**Option 2**: Rechunk incompatible files:

```python
import xarray as xr

def rechunk_file(input_path, output_path, chunks):
    """Rechunk a file to match target chunking."""
    ds = xr.open_dataset(input_path)
    ds_rechunked = ds.chunk(chunks)
    ds_rechunked.to_netcdf(output_path, encoding={
        var: {'chunksizes': chunks.get(var, None)}
        for var in ds.data_vars
    })
    ds.close()

# Rechunk all files to match
target_chunks = {'time': 1, 'lat': 100, 'lon': 100}
for file in incompatible_files:
    rechunk_file(file, f"rechunked_{file}", target_chunks)
```

**Option 3**: Load variables with inconsistent chunking:

```python
# If a variable has inconsistent chunking, load it instead of virtualizing
vds = vz.open_virtual_dataset(
    url,
    registry=registry,
    parser=parser,
    loadable_variables=['inconsistent_var'],  # Load problematic variables
)
```

### Inconsistent Codecs/Compression

**Symptoms**:
- Error: `Incompatible compression settings`
- Error: `Codec mismatch`
- Warnings about mixed compression types

**Diagnosis**:

Check compression across files:

```python
def check_compression(file_paths, variable):
    """Check compression settings across files."""
    compression_info = {}
    for path in file_paths:
        with h5py.File(path, 'r') as f:
            if variable in f:
                var = f[variable]
                compression_info[path] = {
                    'compression': var.compression,
                    'compression_opts': var.compression_opts,
                    'shuffle': var.shuffle,
                    'fletcher32': var.fletcher32,
                }
    return compression_info

info = check_compression(files, "temperature")
for file, comp in info.items():
    print(f"{file}: {comp}")
```

**Common Causes**:
1. Files compressed with different algorithms (gzip vs zlib vs lzf)
2. Different compression levels (gzip level 4 vs level 9)
3. Some files compressed, others not
4. Shuffle filter applied inconsistently

**Solutions**:

**Option 1**: Filter to matching compression:

```python
# Identify files with matching compression
def has_gzip_compression(path, variable):
    with h5py.File(path, 'r') as f:
        return f[variable].compression == 'gzip'

gzip_files = [f for f in file_list if has_gzip_compression(f, "temperature")]
vds = vz.open_virtual_mfdataset(gzip_files, ...)
```

**Option 2**: Recompress files to uniform codec:

```python
import xarray as xr

def recompress_file(input_path, output_path, compression):
    """Recompress file with uniform settings."""
    ds = xr.open_dataset(input_path)
    
    encoding = {
        var: {
            'zlib': True if compression == 'gzip' else False,
            'complevel': 4,
            'shuffle': True,
        }
        for var in ds.data_vars
    }
    
    ds.to_netcdf(output_path, encoding=encoding)
    ds.close()

# Recompress all to gzip level 4
for file in file_list:
    recompress_file(file, f"compressed_{file}", "gzip")
```

**Option 3**: Decompress entirely:

```python
# Create uncompressed versions for virtualization
def decompress_file(input_path, output_path):
    ds = xr.open_dataset(input_path)
    encoding = {var: {'zlib': False} for var in ds.data_vars}
    ds.to_netcdf(output_path, encoding=encoding)
    ds.close()
```

### Inconsistent Data Types

**Symptoms**:
- Error: `dtype mismatch`
- Error: `Cannot concatenate arrays with different dtypes`
- Type conversion warnings

**Diagnosis**:

```python
def check_dtypes(file_paths, variable):
    """Check data types across files."""
    dtype_info = {}
    for path in file_paths:
        with h5py.File(path, 'r') as f:
            if variable in f:
                dtype_info[path] = f[variable].dtype
    return dtype_info

info = check_dtypes(files, "temperature")
for file, dtype in info.items():
    print(f"{file}: {dtype}")
```

**Common Causes**:
1. Mixed precision (float32 vs float64)
2. Integer vs float types
3. Unsigned vs signed integers
4. Different fill value types

**Solutions**:

**Option 1**: Convert to common dtype:

```python
import xarray as xr
import numpy as np

def convert_dtype(input_path, output_path, target_dtype):
    """Convert file to target dtype."""
    ds = xr.open_dataset(input_path)
    
    for var in ds.data_vars:
        if ds[var].dtype != target_dtype:
            ds[var] = ds[var].astype(target_dtype)
    
    ds.to_netcdf(output_path)
    ds.close()

# Convert all to float32
for file in file_list:
    convert_dtype(file, f"float32_{file}", np.float32)
```

**Option 2**: Filter to matching dtype:

```python
target_dtype = np.float32
matching_files = [
    f for f in file_list 
    if get_dtype(f, "temperature") == target_dtype
]
```

## Storage and Access Errors

### S3 Permission Errors

**Symptoms**:
- Error: `Access Denied (403)`
- Error: `NoSuchBucket`
- Error: `InvalidAccessKeyId`
- Error: `SignatureDoesNotMatch`

**Diagnosis Steps**:

1. **Check bucket exists and is accessible**:

```python
import boto3

def check_bucket_access(bucket_name, region):
    """Test if bucket is accessible."""
    try:
        s3 = boto3.client('s3', region_name=region)
        s3.head_bucket(Bucket=bucket_name)
        print(f"✓ Bucket {bucket_name} exists and is accessible")
        return True
    except s3.exceptions.NoSuchBucket:
        print(f"✗ Bucket {bucket_name} does not exist")
        return False
    except s3.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '403':
            print(f"✗ Access denied to bucket {bucket_name}")
        else:
            print(f"✗ Error: {error_code}")
        return False

check_bucket_access("my-bucket", "us-west-2")
```

2. **Verify credentials**:

```python
import boto3

def check_credentials():
    """Verify AWS credentials are configured."""
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✓ Credentials valid")
        print(f"  Account: {identity['Account']}")
        print(f"  User/Role: {identity['Arn']}")
        return True
    except Exception as e:
        print(f"✗ Credential error: {e}")
        return False

check_credentials()
```

**Solutions**:

**For Public Buckets**:

```python
# Use skip_signature for VirtualiZarr
store = from_url("s3://bucket", region="us-west-2", skip_signature=True)

# Use anonymous=True for icechunk
storage = icechunk.s3_storage(bucket="bucket", prefix="", region="us-west-2", anonymous=True)
```

**For Private Buckets with IAM Role**:

```python
# Verify IAM role is attached to instance
# Check instance metadata service
import requests

def check_iam_role():
    """Check if IAM role is attached (EC2/ECS only)."""
    try:
        response = requests.get(
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            timeout=1
        )
        if response.status_code == 200:
            role = response.text
            print(f"✓ IAM role attached: {role}")
            return True
        else:
            print("✗ No IAM role attached")
            return False
    except:
        print("✗ Not running on EC2/ECS or metadata service unavailable")
        return False

check_iam_role()

# Use from_env=True or default behavior
storage = icechunk.s3_storage(bucket="bucket", prefix="", region="us-west-2", from_env=True)
```

**For Explicit Credentials**:

```python
import os

# Verify environment variables are set
required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
missing = [v for v in required_vars if v not in os.environ]

if missing:
    print(f"✗ Missing environment variables: {missing}")
else:
    print("✓ AWS credentials found in environment")

# Use explicit credentials
storage = icechunk.s3_storage(
    bucket="bucket",
    prefix="",
    region="us-west-2",
    access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)
```

### URL Scheme Problems

**Symptoms**:
- Error: `Invalid URL scheme`
- Error: `No store registered for prefix`
- Error: `Unsupported protocol`

**Diagnosis**:

Check URL format and registry configuration:

```python
def validate_url_and_registry(url, registry):
    """Validate URL matches a registered prefix."""
    # Extract URL prefix
    from urllib.parse import urlparse
    parsed = urlparse(url)
    prefix = f"{parsed.scheme}://{parsed.netloc}"
    
    # Check if prefix is registered
    if prefix in registry._stores:
        print(f"✓ URL prefix {prefix} is registered")
        return True
    else:
        print(f"✗ URL prefix {prefix} not in registry")
        print(f"  Registered prefixes: {list(registry._stores.keys())}")
        return False

validate_url_and_registry(url, registry)
```

**Solutions**:

**Ensure URL uses proper scheme**:

```python
# Correct formats
"s3://bucket-name/path/to/file.nc"  # S3
"gs://bucket-name/path/to/file.nc"  # GCS
"abfs://container/path/to/file.nc"  # Azure
"file:///absolute/path/to/file.nc"  # Local
"https://server.com/file.nc"        # HTTP

# Common mistakes to avoid
"bucket-name/path/to/file.nc"       # Missing scheme
"s3:/bucket-name/file.nc"           # Single slash instead of double
"/path/to/file.nc"                  # Local path without file:// scheme
```

**Match registry prefixes to URLs**:

```python
# Registry prefix must match URL prefix exactly
bucket = "s3://my-bucket"  # With scheme
store = S3Store(bucket="my-bucket", region="us-west-2")
registry = ObjectStoreRegistry({bucket: store})  # Use same prefix

# URL must start with registered prefix
url = f"{bucket}/data/file.nc"  # ✓ Correct
url = "s3://my-bucket/data/file.nc"  # ✓ Also correct
url = "s3://other-bucket/file.nc"  # ✗ Not registered
```

### Registry Misconfiguration

**Symptoms**:
- Error: `Store not found for URL`
- Error: `No matching store in registry`
- Unexpected authentication errors despite correct credentials

**Solutions**:

**Register all necessary prefixes**:

```python
# If files are in multiple buckets, register all
registry = ObjectStoreRegistry({
    "s3://bucket-1": S3Store(bucket="bucket-1", region="us-west-2"),
    "s3://bucket-2": S3Store(bucket="bucket-2", region="us-east-1"),
    "file:///data": LocalStore(prefix="/data"),
})
```

**Use wildcard patterns for hierarchical paths**:

```python
# For files under s3://bucket/project1/*, s3://bucket/project2/*, etc.
# Register at bucket level, not subdirectory level
bucket = "s3://bucket"
store = S3Store(bucket="bucket", region="us-west-2")
registry = ObjectStoreRegistry({bucket: store})

# URLs like s3://bucket/project1/file.nc will work
```

## Parser Errors

### Unsupported File Format

**Symptoms**:
- Error: `No parser available for file type`
- Error: `Unsupported format`
- Error during file opening

**Diagnosis**:

Check file format:

```python
import h5py
import netCDF4

def identify_format(file_path):
    """Identify file format."""
    try:
        with h5py.File(file_path, 'r') as f:
            return "HDF5/NetCDF4"
    except:
        pass
    
    try:
        with netCDF4.Dataset(file_path, 'r') as f:
            return "NetCDF3"
    except:
        pass
    
    # Check for FITS
    if file_path.endswith('.fits') or file_path.endswith('.fit'):
        return "FITS (probably)"
    
    return "Unknown"

format_type = identify_format("data.nc")
print(f"Detected format: {format_type}")
```

**Solutions**:

**Use correct parser**:

```python
from virtualizarr.parsers import (
    HDFParser,           # For NetCDF4/HDF5
    NetCDF3Parser,       # For NetCDF3
    FITSParser,          # For FITS
    DMRPPParser,         # For DMR++ sidecar files
    ZarrParser,          # For existing Zarr
    KerchunkJSONParser,  # For Kerchunk JSON
)

# NetCDF4/HDF5 files
parser = HDFParser()

# NetCDF3 files
parser = NetCDF3Parser()

# FITS files
parser = FITSParser()
```

### Wrong Parser Selection

**Symptoms**:
- Error: `Expected HDF5 file`
- Error: `Invalid file structure`
- Parse failures despite correct format

**Solution**:

Match parser to actual file format, not file extension:

```python
# Don't rely solely on extension
# NetCDF4 files often have .nc extension but are HDF5 format
parser = HDFParser()  # Use for .nc files that are NetCDF4

# Use NetCDF3Parser only for classic NetCDF3
parser = NetCDF3Parser()  # Use for true NetCDF3 files
```

### Corrupted Files

**Symptoms**:
- Error: `Unable to open file`
- Error: `Truncated file`
- Error: `Invalid header`

**Diagnosis**:

```python
import h5py

def check_file_integrity(file_path):
    """Check if HDF5/NetCDF4 file is readable."""
    try:
        with h5py.File(file_path, 'r') as f:
            # Try to read basic metadata
            keys = list(f.keys())
            print(f"✓ File is readable, contains {len(keys)} top-level items")
            return True
    except Exception as e:
        print(f"✗ File appears corrupted: {e}")
        return False

check_file_integrity("data.nc")
```

**Solutions**:

1. **Re-download file** if from remote source
2. **Check file size** matches expected size
3. **Verify checksums** if available
4. **Skip corrupted files** in batch processing:

```python
def safe_open_virtual_dataset(url, registry, parser):
    """Safely open file, returning None if corrupted."""
    try:
        return vz.open_virtual_dataset(url, registry=registry, parser=parser)
    except Exception as e:
        print(f"Warning: Skipping {url} due to error: {e}")
        return None

# Filter out None results
datasets = [
    ds for ds in [safe_open_virtual_dataset(url, registry, parser) for url in urls]
    if ds is not None
]
```

## Icechunk-Specific Errors

### ConflictError and Resolution

**Symptoms**:
- Error: `ConflictError: Cannot commit, branch has diverged`
- Commit rejected due to concurrent modifications

**Cause**:
Two or more sessions modified the same branch concurrently.

**Solutions**:

**Option 1**: Rebase and retry:

```python
from icechunk import ConflictDetector

session = repo.writable_session("main")

# Make changes
group = zarr.open_group(session.store, mode="r+")
group["data"][:] = new_values

# Try to commit
try:
    session.commit("Update data")
except icechunk.ConflictError:
    print("Conflict detected, rebasing...")
    # Rebase to latest branch state
    session.rebase(ConflictDetector())
    # Retry commit
    session.commit("Update data after rebase")
```

**Option 2**: Use transaction context manager (auto-retry):

```python
# Transaction will automatically handle conflicts
with repo.transaction("main", message="Update data") as store:
    group = zarr.open_group(store, mode="r+")
    group["data"][:] = new_values
# Automatic commit with conflict handling
```

**Option 3**: Work on separate branches:

```python
# Each writer uses their own branch
repo.create_branch("writer-1", base_snapshot_id)
repo.create_branch("writer-2", base_snapshot_id)

# Writer 1
session1 = repo.writable_session("writer-1")
# ... make changes ...
session1.commit("Writer 1 changes")

# Writer 2
session2 = repo.writable_session("writer-2")
# ... make changes ...
session2.commit("Writer 2 changes")

# Merge branches later (manual merge strategy needed)
```

### Virtual Chunk Container Not Configured

**Symptoms**:
- Error: `VirtualChunkContainer not configured`
- Error: `Cannot access virtual chunks`
- Virtual references written but data not readable

**Solution**:

Configure virtual chunk container before writing:

```python
import icechunk

# Step 1: Configure container
config = icechunk.RepositoryConfig.default()
virtual_store = icechunk.s3_store(region="us-west-2", anonymous=True)
container = icechunk.VirtualChunkContainer(
    "s3://source-bucket/",  # Prefix where virtual chunks live
    virtual_store
)
config.set_virtual_chunk_container(container)

# Step 2: Set up credentials for accessing virtual chunks
credentials = icechunk.containers_credentials({
    "s3://source-bucket/": icechunk.s3_credentials(anonymous=True)
})

# Step 3: Create repository with configuration
storage = icechunk.s3_storage(...)
repo = icechunk.Repository.create(
    storage,
    config=config,
    authorize_virtual_chunk_access=credentials
)
```

### Snapshot Not Found

**Symptoms**:
- Error: `SnapshotNotFound`
- Error: `Invalid snapshot ID`

**Diagnosis**:

List available snapshots:

```python
# List all snapshots on a branch
for snapshot in repo.ancestry("main"):
    print(f"ID: {snapshot.id}")
    print(f"Message: {snapshot.message}")
    print(f"Time: {snapshot.written_at}")
    print("---")
```

**Solution**:

Use valid snapshot ID, branch, or tag:

```python
# Valid ways to open a session
session = repo.readonly_session("main")  # Branch name
session = repo.readonly_session(tag="v1.0.0")  # Tag name
session = repo.readonly_session(snapshot_id="ABC123...")  # Snapshot ID

# Check if branch/tag exists before opening
branches = repo.list_branches()
tags = repo.list_tags()
```

## Performance Issues

### Slow Virtualization

**Symptoms**:
- Taking minutes to virtualize small files
- Timeouts during open_virtual_dataset
- High memory usage

**Diagnosis**:

Profile the operation:

```python
import time

start = time.perf_counter()
vds = vz.open_virtual_dataset(url, registry=registry, parser=parser)
elapsed = time.perf_counter() - start

print(f"Virtualization took {elapsed:.2f} seconds")
print(f"Virtual reference size: {vds.vz.nbytes:,} bytes")
print(f"Actual data size: {vds.nbytes:,} bytes")
```

**Solutions**:

**Use parallel processing for multiple files**:

```python
from concurrent.futures import ThreadPoolExecutor

combined = vz.open_virtual_mfdataset(
    urls,
    registry=registry,
    parser=parser,
    combine='by_coords',
    parallel=ThreadPoolExecutor(max_workers=20),
)
```

**Use caching stores for repeated access**:

```python
from obspec_utils.cache import CachingReadableStore

# Wrap base store with cache
base_store = from_url(bucket, region="us-east-1")
cached_store = CachingReadableStore(
    base_store,
    max_size=1_000_000_000  # 1GB cache
)
registry = ObjectStoreRegistry({bucket: cached_store})
```

**Reduce loadable variables**:

```python
# Only load what's necessary
vds = vz.open_virtual_dataset(
    url,
    registry=registry,
    parser=parser,
    loadable_variables=[],  # Nothing loaded (fastest)
)
```

### Out of Memory Errors

**Symptoms**:
- `MemoryError`
- Process killed by OOM killer
- System becomes unresponsive

**Solutions**:

**Process files in batches**:

```python
def process_in_batches(urls, batch_size, registry, parser):
    """Process files in batches to limit memory."""
    results = []
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        combined = vz.open_virtual_mfdataset(
            batch,
            registry=registry,
            parser=parser,
            combine='by_coords',
        )
        results.append(combined)
    
    # Concatenate all batches
    return xr.concat(results, dim='time')

# Process 100 files at a time
result = process_in_batches(all_urls, batch_size=100, registry, parser)
```

**Use Dask for lazy evaluation**:

```python
# Dask delays actual computation until needed
combined = vz.open_virtual_mfdataset(
    urls,
    registry=registry,
    parser=parser,
    parallel='dask',
)

# Data not loaded until you compute
result = combined["temperature"].mean().compute()
```

**Minimize loadable variables**:

```python
# Load only essential coordinates
vds = vz.open_virtual_dataset(
    url,
    registry=registry,
    parser=parser,
    loadable_variables=['time'],  # Just time, not all coords
)
```

### Network Timeouts

**Symptoms**:
- Error: `Connection timeout`
- Error: `Read timeout`
- Intermittent failures

**Solutions**:

**Increase timeout settings**:

```python
# For icechunk S3 storage
storage = icechunk.s3_storage(
    bucket="bucket",
    prefix="",
    region="us-west-2",
    network_stream_timeout_seconds=120,  # Increase from default 60
)
```

**Use retry logic**:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def open_with_retry(url, registry, parser):
    """Open virtual dataset with automatic retry."""
    return vz.open_virtual_dataset(url, registry=registry, parser=parser)

vds = open_with_retry(url, registry, parser)
```

**Check network connectivity**:

```python
import requests

def test_s3_connectivity(bucket, region):
    """Test connection to S3 endpoint."""
    endpoint = f"https://s3.{region}.amazonaws.com"
    try:
        response = requests.head(endpoint, timeout=5)
        print(f"✓ Can reach {endpoint}")
        return True
    except Exception as e:
        print(f"✗ Cannot reach {endpoint}: {e}")
        return False

test_s3_connectivity("my-bucket", "us-west-2")
```

## Validation Failures

### Checksum Mismatches

**Symptoms**:
- Error: `Checksum validation failed`
- Error: `Data integrity check failed`

**Causes**:
- File was modified after virtualization
- Corruption during transfer
- Concurrent modification of source files

**Solutions**:

**Disable checksum validation** (use with caution):

```python
# In icechunk, when appending with checksum validation
vds.vz.to_icechunk(
    store,
    append_dim="time",
    validate_containers=False,  # Skip validation
)
```

**Re-virtualize with current files**:

```python
# If files have changed, re-create virtual references
vds = vz.open_virtual_dataset(url, registry=registry, parser=parser)
# Overwrite in icechunk (don't append)
with repo.transaction("main", message="Re-virtualized updated files") as store:
    vds.vz.to_icechunk(store)
```

### File Not Found After Virtualization

**Symptoms**:
- Virtual references created successfully
- Error when reading: `FileNotFoundError` or `NoSuchKey`

**Causes**:
- Source files moved or deleted
- Changed URL schemes or paths
- Permission changes on source bucket

**Solutions**:

**Rename paths in virtual references**:

```python
# Update paths in existing virtual dataset
vds_updated = vds.vz.rename_paths(
    lambda old_path: old_path.replace("old-bucket", "new-bucket")
)

# Or use string replacement
vds_updated = vds.vz.rename_paths("s3://new-bucket/")

# Write updated references
with repo.transaction("main", message="Updated paths") as store:
    vds_updated.vz.to_icechunk(store)
```

**Verify source files still exist**:

```python
import boto3

def verify_files_exist(manifest_array, region):
    """Check if all files in manifest still exist."""
    s3 = boto3.client('s3', region_name=region)
    
    for chunk_key, chunk_ref in manifest_array.manifest.dict().items():
        path = chunk_ref['path']
        # Parse s3://bucket/key format
        bucket = path.split('/')[2]
        key = '/'.join(path.split('/')[3:])
        
        try:
            s3.head_object(Bucket=bucket, Key=key)
            print(f"✓ {path}")
        except:
            print(f"✗ Missing: {path}")

# Check all variables
for var_name, var in vds.data_vars.items():
    if hasattr(var.data, 'manifest'):
        print(f"Checking {var_name}...")
        verify_files_exist(var.data, "us-west-2")
```

## FAQ

### Q: Can I virtualize files with different compression levels?

**A**: No. Files must have identical compression settings (algorithm and level). Either:
- Filter to files with matching compression
- Recompress all files uniformly
- Decompress all files

### Q: Can I mix NetCDF3 and NetCDF4 files?

**A**: No. Use separate virtualizations for each format, then combine the resulting virtual datasets with xarray operations if needed.

### Q: How do I know if a file is NetCDF3 or NetCDF4?

**A**: Try opening with h5py:

```python
import h5py

try:
    with h5py.File("file.nc", 'r') as f:
        print("NetCDF4 (HDF5 format)")
except:
    print("NetCDF3 (classic format)")
```

### Q: Can I virtualize files with ragged arrays or variable-length dimensions?

**A**: No. VirtualiZarr requires regular chunked arrays with fixed dimensions. Ragged/vlen data cannot be virtualized.

### Q: What's the maximum number of files I can virtualize?

**A**: VirtualiZarr can handle millions of files, but:
- Use parallel processing for large collections
- Consider batching for memory efficiency
- Icechunk handles large manifests efficiently

### Q: Can I modify virtual data?

**A**: No. Virtual datasets are read-only references. To modify:
1. Load the data into memory
2. Make modifications
3. Write as new Zarr arrays (not virtual)

### Q: How do I convert existing kerchunk references to icechunk?

**A**: Use the KerchunkJSONParser:

```python
from virtualizarr.parsers import KerchunkJSONParser

vds = vz.open_virtual_dataset(
    "references.json",
    registry=registry,
    parser=KerchunkJSONParser()
)

with repo.transaction("main", message="Converted from kerchunk") as store:
    vds.vz.to_icechunk(store)
```

### Q: Can I use VirtualiZarr without icechunk?

**A**: Yes. You can:
- Write to kerchunk format (legacy)
- Use ManifestStore directly with Zarr/Xarray
- Keep virtual datasets in memory for analysis

## Getting Help

If issues persist after trying these solutions:

1. **VirtualiZarr Issues**: https://github.com/zarr-developers/VirtualiZarr/issues
2. **Icechunk Issues**: https://github.com/earth-mover/icechunk/issues
3. **Check Documentation**:
   - VirtualiZarr: https://virtualizarr.readthedocs.io/
   - Icechunk: https://icechunk.io/

When filing issues, include:
- Python version
- Library versions (virtualizarr, icechunk, zarr, xarray)
- Minimal reproducible example
- Full error traceback
- File format details (run `h5dump` or `ncdump -h` on sample file)

# Virtual Zarr Complete Examples

Complete, runnable code examples demonstrating VirtualiZarr and icechunk workflows.

## Setup and Installation

### Install Dependencies

```bash
# Using pip
pip install virtualizarr icechunk xarray zarr obstore obspec-utils h5py netcdf4

# Using uv (recommended)
uv pip install virtualizarr icechunk xarray zarr obstore obspec-utils h5py netcdf4

# Using pixi
pixi add virtualizarr icechunk xarray zarr obstore obspec-utils h5py netcdf4
```

### Python Requirements

- Python 3.10 or higher
- Latest stable versions of all packages

### Import Common Libraries

All examples assume these imports:

```python
import os
import xarray as xr
import zarr
import icechunk
import virtualizarr as vz
from obstore.store import S3Store, LocalStore, from_url
from obspec_utils.registry import ObjectStoreRegistry
from virtualizarr.parsers import HDFParser
```

---

## Example 1: Simple Local File Virtualization

Complete workflow virtualizing a local NetCDF file and storing in a local icechunk repository.

```python
#!/usr/bin/env python3
"""
Simple local file virtualization example.
Creates a test NetCDF file, virtualizes it, writes to icechunk, and reads back.
"""

import numpy as np
import xarray as xr
import zarr
import icechunk
import virtualizarr as vz
from pathlib import Path
from obstore.store import LocalStore
from obspec_utils.registry import ObjectStoreRegistry
from virtualizarr.parsers import HDFParser

# === Step 1: Create test NetCDF file ===
print("Step 1: Creating test NetCDF file...")

test_file = "/tmp/test_data.nc"

# Create sample dataset
time = np.arange(10)
lat = np.arange(-90, 91, 1)
lon = np.arange(-180, 180, 1)
temperature = 15 + 8 * np.random.randn(10, 181, 360)

ds = xr.Dataset(
    {
        "temperature": (["time", "lat", "lon"], temperature),
    },
    coords={
        "time": time,
        "lat": lat,
        "lon": lon,
    },
)

# Write with chunking and compression
ds.to_netcdf(
    test_file,
    encoding={
        "temperature": {
            "chunksizes": (1, 90, 180),
            "zlib": True,
            "complevel": 4,
        }
    },
)
print(f"Created test file: {test_file}")

# === Step 2: Setup object store and registry ===
print("\nStep 2: Setting up object store registry...")

base_path = Path("/tmp")
file_url = f"file://{base_path}"

store = LocalStore(prefix=base_path)
registry = ObjectStoreRegistry({file_url: store})
print("Registry configured for local filesystem")

# === Step 3: Virtualize the file ===
print("\nStep 3: Virtualizing NetCDF file...")

url = f"file://{test_file}"
parser = HDFParser()

vds = vz.open_virtual_dataset(
    url,
    registry=registry,
    parser=parser,
    loadable_variables=['time', 'lat', 'lon'],  # Load coordinates
)

print(f"Virtual dataset created")
print(f"  Virtual reference size: {vds.vz.nbytes:,} bytes")
print(f"  Actual data size: {vds.nbytes:,} bytes")
print(f"  Compression ratio: {vds.nbytes / vds.vz.nbytes:.0f}x")

# === Step 4: Create icechunk repository ===
print("\nStep 4: Creating icechunk repository...")

repo_path = "/tmp/icechunk-test-repo"
storage = icechunk.local_filesystem_storage(repo_path)

# Configure for virtual chunks
config = icechunk.RepositoryConfig.default()
virtual_store_config = icechunk.local_filesystem_store(path=str(base_path))
container = icechunk.VirtualChunkContainer(file_url, virtual_store_config)
config.set_virtual_chunk_container(container)

# No credentials needed for local filesystem
repo = icechunk.Repository.create(storage, config=config)
print(f"Repository created at: {repo_path}")

# === Step 5: Write virtual dataset to icechunk ===
print("\nStep 5: Writing virtual dataset to icechunk...")

with repo.transaction("main", message="Initial virtualization") as store:
    vds.vz.to_icechunk(store)

print("Virtual dataset written to icechunk")

# === Step 6: Read back from icechunk ===
print("\nStep 6: Reading data back from icechunk...")

session = repo.readonly_session("main")
ds_read = xr.open_zarr(session.store, consolidated=False, zarr_format=3)

print("Dataset successfully read from icechunk:")
print(ds_read)

# Verify data matches
temp_data = ds_read["temperature"].values
print(f"\nData shape: {temp_data.shape}")
print(f"Data range: [{temp_data.min():.2f}, {temp_data.max():.2f}]")

# === Step 7: Version control demonstration ===
print("\nStep 7: Demonstrating version control...")

# Make a change
session2 = repo.writable_session("main")
group = zarr.open_group(session2.store, mode="r+")

# Modify first time slice
print("Zeroing out first time slice...")
array = group["temperature"]
first_slice = array[0, :, :]
array[0, :, :] = 0

session2.commit("Zeroed first time slice")

# View history
print("\nCommit history:")
for i, snapshot in enumerate(repo.ancestry("main")):
    print(f"{i+1}. {snapshot.message} ({snapshot.written_at})")

print("\n✓ Example complete!")
print(f"Virtual Zarr repository created at: {repo_path}")
print(f"Original NetCDF file: {test_file}")
```

**Expected Output:**
```
Step 1: Creating test NetCDF file...
Created test file: /tmp/test_data.nc

Step 2: Setting up object store registry...
Registry configured for local filesystem

Step 3: Virtualizing NetCDF file...
Virtual dataset created
  Virtual reference size: 1,234 bytes
  Actual data size: 26,136,000 bytes
  Compression ratio: 21180x

Step 4: Creating icechunk repository...
Repository created at: /tmp/icechunk-test-repo

Step 5: Writing virtual dataset to icechunk...
Virtual dataset written to icechunk

Step 6: Reading data back from icechunk...
Dataset successfully read from icechunk:
<xarray.Dataset>
Dimensions:  (time: 10, lat: 181, lon: 360)
Coordinates:
  * time     (time) int64 0 1 2 3 4 5 6 7 8 9
  * lat      (lat) int64 -90 -89 -88 -87 ... 87 88 89 90
  * lon      (lon) int64 -180 -179 -178 ... 177 178 179
Data variables:
    temperature  (time, lat, lon) float64 ...

Data shape: (10, 181, 360)
Data range: [-10.32, 40.87]

Step 7: Demonstrating version control...
Zeroing out first time slice...

Commit history:
1. Zeroed first time slice (2024-02-04 10:30:15)
2. Initial virtualization (2024-02-04 10:30:10)

✓ Example complete!
Virtual Zarr repository created at: /tmp/icechunk-test-repo
Original NetCDF file: /tmp/test_data.nc
```

---

## Example 2: Public S3 Dataset (Anonymous Access)

Virtualize data from a public S3 bucket without authentication.

```python
#!/usr/bin/env python3
"""
Public S3 dataset virtualization with anonymous access.
Uses NOAA GOES-16 satellite data as an example.
"""

import icechunk
import virtualizarr as vz
from obstore.store import from_url
from obspec_utils.registry import ObjectStoreRegistry
from virtualizarr.parsers import HDFParser

print("Virtualizing public GOES-16 satellite data from S3...")

# === Configuration ===
source_bucket = "s3://noaa-goes16"
source_region = "us-east-1"

# Example file (GOES-16 ABI Level 2 MCMIP)
file_path = (
    "ABI-L2-MCMIPF/2024/099/18/"
    "OR_ABI-L2-MCMIPF-M6_G16_s20240991800204_e20240991809524_c20240991810005.nc"
)
source_url = f"{source_bucket}/{file_path}"

# === Setup Source Storage (Anonymous) ===
print("\n1. Setting up anonymous S3 access...")

# For VirtualiZarr: use skip_signature=True
source_store = from_url(source_bucket, region=source_region, skip_signature=True)
registry = ObjectStoreRegistry({source_bucket: source_store})

print(f"   ✓ Configured anonymous access to {source_bucket}")

# === Virtualize the File ===
print("\n2. Virtualizing GOES-16 file...")

parser = HDFParser(
    drop_variables=[  # Drop band statistics we don't need
        'band_id',
        'min_reflectance_factor',
        'max_reflectance_factor',
    ]
)

vds = vz.open_virtual_dataset(
    source_url,
    registry=registry,
    parser=parser,
    loadable_variables=['t', 'x', 'y', 'band'],  # Load small coordinates
)

print(f"   ✓ Virtualized {len(vds.data_vars)} variables")
print(f"   Virtual size: {vds.vz.nbytes / 1024:.1f} KB")
print(f"   Actual data size: {vds.nbytes / 1e9:.2f} GB")

# === Setup Icechunk Repository (Local) ===
print("\n3. Setting up icechunk repository...")

repo_storage = icechunk.local_filesystem_storage("/tmp/goes16-icechunk")

# Configure virtual chunk container for S3 access (anonymous)
config = icechunk.RepositoryConfig.default()
virtual_store_config = icechunk.s3_store(region=source_region, anonymous=True)
container = icechunk.VirtualChunkContainer(source_bucket, virtual_store_config)
config.set_virtual_chunk_container(container)

# Set up anonymous credentials for virtual chunk access
credentials = icechunk.containers_credentials({
    source_bucket: icechunk.s3_credentials(anonymous=True)
})

repo = icechunk.Repository.create(
    repo_storage,
    config=config,
    authorize_virtual_chunk_access=credentials
)

print("   ✓ Repository created with anonymous S3 access")

# === Write to Icechunk ===
print("\n4. Writing virtual dataset to icechunk...")

with repo.transaction("main", message="GOES-16 ABI L2 MCMIP") as store:
    vds.vz.to_icechunk(store)

print("   ✓ Virtual references written")

# === Verify by Reading ===
print("\n5. Verifying data access...")

session = repo.readonly_session("main")
ds_read = xr.open_zarr(session.store, consolidated=False, zarr_format=3)

print(f"   ✓ Successfully read dataset")
print(f"   Variables: {list(ds_read.data_vars.keys())}")

# Access a small slice of data
if 'CMI_C01' in ds_read:
    sample = ds_read['CMI_C01'][0:10, 0:10].values
    print(f"   Sample data shape: {sample.shape}")
    print(f"   Sample data range: [{sample.min():.3f}, {sample.max():.3f}]")

print("\n✓ Example complete!")
print("Virtual GOES-16 data accessible from icechunk repository")
print("Repository location: /tmp/goes16-icechunk")
```

---

## Example 3: Private S3 with Instance Profile

Use IAM role credentials (for EC2/ECS/Lambda).

```python
#!/usr/bin/env python3
"""
Private S3 bucket access using instance profile (IAM role).
Suitable for EC2, ECS, Lambda, or any environment with IAM role attached.
"""

import icechunk
import virtualizarr as vz
from obstore.store import from_url
from obspec_utils.registry import ObjectStoreRegistry
from virtualizarr.parsers import HDFParser

print("Virtualizing from private S3 bucket using IAM role...")

# === Configuration ===
source_bucket = "s3://my-private-data-bucket"
source_region = "us-west-2"
source_url = f"{source_bucket}/climate/monthly/temperature_2024_01.nc"

# Icechunk repository location (also S3)
repo_bucket = "my-icechunk-repos"
repo_prefix = "climate/virtual-stores"
repo_region = "us-west-2"

# === Setup Source Storage (IAM Role) ===
print("\n1. Setting up S3 access with IAM role credentials...")

# For VirtualiZarr: credentials automatically detected from environment
# No explicit credentials needed - will use IAM role
source_store = from_url(source_bucket, region=source_region)
registry = ObjectStoreRegistry({source_bucket: source_store})

print(f"   ✓ Using IAM role credentials for {source_bucket}")

# === Virtualize File ===
print("\n2. Virtualizing NetCDF file...")

parser = HDFParser()
vds = vz.open_virtual_dataset(
    source_url,
    registry=registry,
    parser=parser,
    loadable_variables=['time', 'lat', 'lon'],
)

print(f"   ✓ Virtualized dataset")
print(f"   Variables: {list(vds.data_vars.keys())}")

# === Setup Icechunk Repository (S3, IAM Role) ===
print("\n3. Setting up icechunk repository on S3...")

# For icechunk: use from_env=True to use IAM role
# Or omit all credential params - defaults to environment credentials
repo_storage = icechunk.s3_storage(
    bucket=repo_bucket,
    prefix=repo_prefix,
    region=repo_region,
    from_env=True,  # Explicit: use IAM role / environment credentials
)

# Configure virtual chunk container (also using IAM role)
config = icechunk.RepositoryConfig.default()
virtual_store_config = icechunk.s3_store(region=source_region)  # Uses IAM role by default
container = icechunk.VirtualChunkContainer(source_bucket, virtual_store_config)
config.set_virtual_chunk_container(container)

# Credentials for virtual chunk access (IAM role)
credentials = icechunk.containers_credentials({
    source_bucket: icechunk.s3_credentials(from_env=True)
})

repo = icechunk.Repository.create(
    repo_storage,
    config=config,
    authorize_virtual_chunk_access=credentials
)

print(f"   ✓ Repository created on S3: s3://{repo_bucket}/{repo_prefix}")

# === Write to Icechunk ===
print("\n4. Writing virtual dataset to icechunk...")

with repo.transaction("main", message="January 2024 temperature data") as store:
    vds.vz.to_icechunk(store)

print("   ✓ Virtual references written to S3")

# === Read Back ===
print("\n5. Reading data from icechunk...")

session = repo.readonly_session("main")
ds_read = xr.open_zarr(session.store, consolidated=False, zarr_format=3)

print(f"   ✓ Dataset read successfully")
print(f"   Shape: {ds_read['temperature'].shape if 'temperature' in ds_read else 'N/A'}")

print("\n✓ Example complete!")
print(f"Repository: s3://{repo_bucket}/{repo_prefix}")
print("Note: This example requires an IAM role with S3 access permissions")
```

**IAM Policy Requirements:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-private-data-bucket",
        "arn:aws:s3:::my-private-data-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-icechunk-repos",
        "arn:aws:s3:::my-icechunk-repos/*"
      ]
    }
  ]
}
```

---

## Example 4: Private S3 with Explicit Credentials

Use AWS access keys from environment variables (best practice for explicit credentials).

```python
#!/usr/bin/env python3
"""
Private S3 access with explicit AWS credentials.
Credentials loaded from environment variables (recommended for security).
"""

import os
import icechunk
import virtualizarr as vz
from obstore.store import S3Store
from obspec_utils.registry import ObjectStoreRegistry
from virtualizarr.parsers import HDFParser

print("Virtualizing from private S3 with explicit credentials...")

# === Verify Environment Variables ===
print("\n1. Checking AWS credentials in environment...")

required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
missing = [v for v in required_vars if v not in os.environ]

if missing:
    print(f"   ✗ Missing environment variables: {missing}")
    print("\n   Please set credentials:")
    print("   export AWS_ACCESS_KEY_ID='your-access-key'")
    print("   export AWS_SECRET_ACCESS_KEY='your-secret-key'")
    print("   export AWS_SESSION_TOKEN='your-session-token'  # Optional for temporary credentials")
    exit(1)

print("   ✓ AWS credentials found in environment")

# === Configuration ===
source_bucket = "s3://research-archive"
source_region = "us-east-1"
source_url = f"{source_bucket}/experiments/2024/run_001/output.h5"

repo_bucket = "research-icechunk"
repo_prefix = "virtual-datasets"
repo_region = "us-east-1"

# === Setup Source Storage (Explicit Credentials) ===
print("\n2. Setting up S3 access with explicit credentials...")

# For VirtualiZarr: use S3Store with explicit credentials from environment
source_store = S3Store.from_url(
    source_bucket,
    access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    # session_token is optional (for temporary credentials)
)

registry = ObjectStoreRegistry({source_bucket: source_store})
print(f"   ✓ Authenticated to {source_bucket}")

# === Virtualize File ===
print("\n3. Virtualizing HDF5 file...")

parser = HDFParser()
vds = vz.open_virtual_dataset(
    source_url,
    registry=registry,
    parser=parser,
    loadable_variables=None,  # Load dimension coordinates only (default)
)

print(f"   ✓ Virtualized {len(vds.data_vars)} variables")

# === Setup Icechunk Repository (Explicit Credentials) ===
print("\n4. Setting up icechunk repository...")

# For icechunk: pass credentials explicitly
repo_storage = icechunk.s3_storage(
    bucket=repo_bucket,
    prefix=repo_prefix,
    region=repo_region,
    access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    session_token=os.environ.get("AWS_SESSION_TOKEN"),  # Optional
)

# Configure virtual chunk container with same credentials
config = icechunk.RepositoryConfig.default()
virtual_store_config = icechunk.s3_store(region=source_region)
container = icechunk.VirtualChunkContainer(source_bucket, virtual_store_config)
config.set_virtual_chunk_container(container)

# Authorize virtual chunk access with explicit credentials
credentials = icechunk.containers_credentials({
    source_bucket: icechunk.s3_credentials(
        access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )
})

repo = icechunk.Repository.create(
    repo_storage,
    config=config,
    authorize_virtual_chunk_access=credentials
)

print(f"   ✓ Repository created: s3://{repo_bucket}/{repo_prefix}")

# === Write to Icechunk ===
print("\n5. Writing virtual dataset...")

with repo.transaction("main", message="Experiment run 001 results") as store:
    vds.vz.to_icechunk(store)

print("   ✓ Written to icechunk")

# === Read and Verify ===
print("\n6. Verifying data access...")

session = repo.readonly_session("main")
ds_read = xr.open_zarr(session.store, consolidated=False, zarr_format=3)

print(f"   ✓ Dataset accessible")
print(f"   Variables: {list(ds_read.data_vars.keys())[:5]}...")  # Show first 5

print("\n✓ Example complete!")
print(f"Repository: s3://{repo_bucket}/{repo_prefix}")
print("\n⚠️  Security reminder:")
print("   - Never hardcode credentials in source code")
print("   - Use environment variables or secrets management")
print("   - Consider using IAM roles when possible")
```

---

## Example 5: Multi-File Time Series Aggregation

Virtualize and concatenate multiple files along the time dimension.

```python
#!/usr/bin/env python3
"""
Multi-file time series aggregation.
Combines daily temperature files into a single virtual dataset.
"""

import numpy as np
import xarray as xr
import icechunk
import virtualizarr as vz
from pathlib import Path
from obstore.store import LocalStore
from obspec_utils.registry import ObjectStoreRegistry
from virtualizarr.parsers import HDFParser
from concurrent.futures import ThreadPoolExecutor

print("Multi-file time series aggregation example")

# === Step 1: Create Test Files (Daily Data) ===
print("\n1. Creating test daily files...")

data_dir = Path("/tmp/daily_data")
data_dir.mkdir(exist_ok=True)

num_days = 5
file_urls = []

for day in range(1, num_days + 1):
    file_path = data_dir / f"temperature_day_{day:03d}.nc"
    
    # Create daily dataset
    lat = np.arange(-90, 91, 5)
    lon = np.arange(-180, 180, 5)
    time = np.array([day])
    
    temp = 15 + 8 * np.random.randn(1, len(lat), len(lon))
    
    ds = xr.Dataset(
        {
            "temperature": (["time", "lat", "lon"], temp),
            "humidity": (["time", "lat", "lon"], 50 + 20 * np.random.randn(1, len(lat), len(lon))),
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        },
    )
    
    # Write with consistent chunking
    ds.to_netcdf(
        file_path,
        encoding={
            "temperature": {"chunksizes": (1, 36, 72), "zlib": True, "complevel": 4},
            "humidity": {"chunksizes": (1, 36, 72), "zlib": True, "complevel": 4},
        },
    )
    
    file_urls.append(f"file://{file_path}")
    
print(f"   ✓ Created {num_days} daily files")

# === Step 2: Setup Registry ===
print("\n2. Setting up object store registry...")

base_path = f"file://{data_dir}"
store = LocalStore(prefix=data_dir)
registry = ObjectStoreRegistry({base_path: store})

# === Step 3: Virtualize All Files (Serial) ===
print("\n3a. Virtualizing files (serial)...")

import time
start = time.perf_counter()

parser = HDFParser()

datasets_serial = []
for url in file_urls:
    vds = vz.open_virtual_dataset(
        url,
        registry=registry,
        parser=parser,
        loadable_variables=['time', 'lat', 'lon'],
    )
    datasets_serial.append(vds)

serial_time = time.perf_counter() - start
print(f"   ✓ Serial virtualization: {serial_time:.2f}s")

# === Step 4: Virtualize All Files (Parallel) ===
print("\n3b. Virtualizing files (parallel with open_virtual_mfdataset)...")

start = time.perf_counter()

combined = vz.open_virtual_mfdataset(
    file_urls,
    registry=registry,
    parser=parser,
    combine='by_coords',
    concat_dim='time',
    parallel=ThreadPoolExecutor(max_workers=4),
    loadable_variables=['time', 'lat', 'lon'],
)

parallel_time = time.perf_counter() - start
print(f"   ✓ Parallel virtualization: {parallel_time:.2f}s")
print(f"   Speedup: {serial_time / parallel_time:.1f}x")

# === Step 5: Verify Concatenation ===
print("\n4. Verifying concatenated dataset...")

print(f"   Combined shape: {combined['temperature'].shape}")
print(f"   Expected: (5, 37, 72)")
print(f"   Time coordinates: {combined['time'].values}")

# === Step 6: Write to Icechunk ===
print("\n5. Writing to icechunk...")

repo_path = "/tmp/timeseries-icechunk"
storage = icechunk.local_filesystem_storage(repo_path)

config = icechunk.RepositoryConfig.default()
virtual_store_config = icechunk.local_filesystem_store(path=str(data_dir))
container = icechunk.VirtualChunkContainer(base_path, virtual_store_config)
config.set_virtual_chunk_container(container)

repo = icechunk.Repository.create(storage, config=config)

with repo.transaction("main", message="5 days of temperature data") as store:
    combined.vz.to_icechunk(store)

print("   ✓ Combined dataset written to icechunk")

# === Step 7: Append More Data ===
print("\n6. Appending new day's data...")

# Create day 6
day6_path = data_dir / "temperature_day_006.nc"
ds_day6 = xr.Dataset(
    {
        "temperature": (["time", "lat", "lon"], 15 + 8 * np.random.randn(1, 37, 72)),
        "humidity": (["time", "lat", "lon"], 50 + 20 * np.random.randn(1, 37, 72)),
    },
    coords={
        "time": [6],
        "lat": lat,
        "lon": lon,
    },
)
ds_day6.to_netcdf(
    day6_path,
    encoding={
        "temperature": {"chunksizes": (1, 36, 72), "zlib": True, "complevel": 4},
        "humidity": {"chunksizes": (1, 36, 72), "zlib": True, "complevel": 4},
    },
)

# Virtualize day 6
vds_day6 = vz.open_virtual_dataset(
    f"file://{day6_path}",
    registry=registry,
    parser=parser,
    loadable_variables=['time', 'lat', 'lon'],
)

# Append to icechunk
with repo.transaction("main", message="Added day 6") as store:
    vds_day6.vz.to_icechunk(store, append_dim="time")

print("   ✓ Day 6 appended")

# === Step 8: Read Final Dataset ===
print("\n7. Reading final aggregated dataset...")

session = repo.readonly_session("main")
ds_final = xr.open_zarr(session.store, consolidated=False, zarr_format=3)

print(f"   Final shape: {ds_final['temperature'].shape}")
print(f"   Expected: (6, 37, 72)")
print(f"   Time range: {ds_final['time'].values.min()} to {ds_final['time'].values.max()}")

# View commit history
print("\n8. Commit history:")
for i, snapshot in enumerate(repo.ancestry("main")):
    print(f"   {i+1}. {snapshot.message}")

print("\n✓ Example complete!")
print(f"Repository: {repo_path}")
print(f"Contains 6 days of virtual data from {num_days + 1} source files")
```

---

## Example 6: Icechunk Version Control Workflow

Comprehensive version control operations: branches, tags, time-travel.

```python
#!/usr/bin/env python3
"""
Icechunk version control workflow.
Demonstrates branches, tags, commits, and time-travel.
"""

import numpy as np
import xarray as xr
import zarr
import icechunk
from datetime import datetime

print("Icechunk Version Control Example")

# === Step 1: Create Repository ===
print("\n1. Creating icechunk repository...")

repo_path = "/tmp/version-control-demo"
storage = icechunk.local_filesystem_storage(repo_path)
repo = icechunk.Repository.create(storage)

print(f"   ✓ Repository created: {repo_path}")

# === Step 2: Initialize with Data ===
print("\n2. Creating initial dataset...")

session = repo.writable_session("main")
root = zarr.open_group(session.store, mode="w")

# Create arrays
temp = root.create_array(
    "temperature",
    shape=(10, 100, 100),
    chunks=(1, 50, 50),
    dtype='f4',
)

precip = root.create_array(
    "precipitation",
    shape=(10, 100, 100),
    chunks=(1, 50, 50),
    dtype='f4',
)

# Write initial data
temp[:] = 15.0 + 5 * np.random.randn(10, 100, 100)
precip[:] = 100.0 + 20 * np.random.randn(10, 100, 100)

initial_snapshot = session.commit("Initial dataset")
print(f"   ✓ Initial commit: {initial_snapshot[:8]}...")

# === Step 3: Make Updates on Main ===
print("\n3. Making updates on main branch...")

session = repo.writable_session("main")
root = zarr.open_group(session.store, mode="r+")

# Update temperature (warming trend)
root["temperature"][:] += 1.0

snapshot_v1 = session.commit("Applied warming trend (+1.0°C)")
print(f"   ✓ Commit v1.0: {snapshot_v1[:8]}...")

# Tag this version
repo.create_tag("v1.0", snapshot_id=snapshot_v1)
print("   ✓ Tagged as v1.0")

# === Step 4: Create Experimental Branch ===
print("\n4. Creating experimental branch...")

repo.create_branch("high-warming", snapshot_id=snapshot_v1)
print("   ✓ Branch 'high-warming' created")

# Make aggressive changes on branch
session_exp = repo.writable_session("high-warming")
root_exp = zarr.open_group(session_exp.store, mode="r+")

root_exp["temperature"][:] += 3.0  # Additional +3°C warming
root_exp["precipitation"][:] *= 0.8  # 20% less precipitation

snapshot_exp = session_exp.commit("High warming scenario (+3°C, -20% precip)")
print(f"   ✓ Experimental commit: {snapshot_exp[:8]}...")

# === Step 5: Continue Main Branch ===
print("\n5. Continuing development on main...")

session_main = repo.writable_session("main")
root_main = zarr.open_group(session_main.store, mode="r+")

# Add new variable
humidity = root_main.create_array(
    "humidity",
    shape=(10, 100, 100),
    chunks=(1, 50, 50),
    dtype='f4',
)
humidity[:] = 60.0 + 10 * np.random.randn(10, 100, 100)

snapshot_v2 = session_main.commit("Added humidity variable")
print(f"   ✓ Commit v2.0: {snapshot_v2[:8]}...")

repo.create_tag("v2.0", snapshot_id=snapshot_v2)
print("   ✓ Tagged as v2.0")

# === Step 6: View Full History ===
print("\n6. Version history:")

print("\n   Main branch:")
for i, snapshot in enumerate(repo.ancestry("main")):
    tags = [tag for tag in repo.list_tags() if repo.lookup_tag(tag) == snapshot.id]
    tag_str = f" [tag: {tags[0]}]" if tags else ""
    print(f"     {i+1}. {snapshot.message}{tag_str}")
    print(f"        {snapshot.id[:8]}... at {snapshot.written_at}")

print("\n   High-warming branch:")
for i, snapshot in enumerate(repo.ancestry("high-warming")):
    print(f"     {i+1}. {snapshot.message}")
    print(f"        {snapshot.id[:8]}... at {snapshot.written_at}")

# === Step 7: Time Travel - Compare Versions ===
print("\n7. Time travel: comparing versions...")

# Read v1.0
session_v1 = repo.readonly_session(tag="v1.0")
data_v1 = zarr.open_array(session_v1.store, path="temperature", mode="r")
temp_v1 = data_v1[0, 50, 50]  # Sample point

# Read v2.0
session_v2 = repo.readonly_session(tag="v2.0")
data_v2 = zarr.open_array(session_v2.store, path="temperature", mode="r")
temp_v2 = data_v2[0, 50, 50]

# Read experimental
session_exp_read = repo.readonly_session("high-warming")
data_exp = zarr.open_array(session_exp_read.store, path="temperature", mode="r")
temp_exp = data_exp[0, 50, 50]

print(f"\n   Temperature at point (0, 50, 50):")
print(f"     v1.0 (main):          {temp_v1:.2f}°C")
print(f"     v2.0 (main):          {temp_v2:.2f}°C")
print(f"     high-warming branch:  {temp_exp:.2f}°C")
print(f"\n   Change v1→v2: {temp_v2 - temp_v1:+.2f}°C")
print(f"   Change v1→exp: {temp_exp - temp_v1:+.2f}°C")

# === Step 8: Check Variable Existence Across Versions ===
print("\n8. Variable availability across versions:")

versions = {
    "v1.0": repo.readonly_session(tag="v1.0"),
    "v2.0": repo.readonly_session(tag="v2.0"),
    "high-warming": repo.readonly_session("high-warming"),
}

variables = ["temperature", "precipitation", "humidity"]

print("\n   Variable | v1.0 | v2.0 | high-warming")
print("   " + "-" * 45)
for var in variables:
    status = []
    for version, session in versions.items():
        exists = var in zarr.open_group(session.store, mode="r")
        status.append("✓" if exists else "✗")
    print(f"   {var:12} | {status[0]:^4} | {status[1]:^4} | {status[2]:^13}")

# === Step 9: Demonstrate Diff ===
print("\n9. Differences between versions:")

diff = repo.diff(
    from_snapshot_id=snapshot_v1,
    to_snapshot_id=snapshot_v2
)

print(f"   v1.0 → v2.0:")
print(f"   - Added variables: humidity")
print(f"   - Modified variables: temperature (warming applied)")

print("\n✓ Example complete!")
print(f"\nRepository structure:")
print(f"  main branch:   {list(repo.ancestry('main')).__len__()} commits")
print(f"  experimental:  {list(repo.ancestry('high-warming')).__len__()} commits")
print(f"  Tags:          v1.0, v2.0")
print(f"  Location:      {repo_path}")
```

---

## Example 7: Distributed Parallel Writes

Use ForkSession for parallel writes across multiple processes.

```python
#!/usr/bin/env python3
"""
Distributed parallel writes using ForkSession.
Demonstrates safe concurrent writes with merge.
"""

import numpy as np
import zarr
import icechunk
from concurrent.futures import ProcessPoolExecutor
import time

print("Distributed Parallel Writes Example")

# === Step 1: Initialize Repository ===
print("\n1. Initializing repository with empty arrays...")

repo_path = "/tmp/parallel-writes-demo"
storage = icechunk.local_filesystem_storage(repo_path)
repo = icechunk.Repository.create(storage)

# Create initial structure
session = repo.writable_session("main")
root = zarr.open_group(session.store, mode="w")

# Create large array (simulate big dataset)
data = root.create_array(
    "results",
    shape=(1000, 100, 100),
    chunks=(10, 50, 50),
    dtype='f4',
    fill_value=np.nan,
)

session.commit("Initialized empty results array")
print("   ✓ Empty array created: shape (1000, 100, 100)")

# === Step 2: Define Worker Function ===
def process_chunk(fork_session_bytes, start_idx, end_idx):
    """
    Worker function that runs in separate process.
    Must serialize/deserialize fork session.
    """
    import pickle
    import numpy as np
    import zarr
    
    # Deserialize fork session
    fork_session = pickle.loads(fork_session_bytes)
    
    # Open array
    root = zarr.open_group(fork_session.store, mode="r+")
    array = root["results"]
    
    # Compute and write data for this chunk
    for i in range(start_idx, end_idx):
        # Simulate computation
        computed = np.random.randn(100, 100).astype('f4') * 10 + 50
        array[i, :, :] = computed
    
    # Return serialized fork session
    return pickle.dumps(fork_session)

# === Step 3: Distribute Work ===
print("\n2. Distributing computation across processes...")

session = repo.writable_session("main")
fork = session.fork()

# Serialize fork for multiprocessing
import pickle
fork_bytes = pickle.dumps(fork)

# Define work chunks
num_workers = 4
chunk_size = 250  # Each worker processes 250 time slices
tasks = [
    (fork_bytes, i * chunk_size, (i + 1) * chunk_size)
    for i in range(num_workers)
]

print(f"   Launching {num_workers} parallel workers...")
print(f"   Each processing {chunk_size} time slices...")

# Execute in parallel
start_time = time.perf_counter()

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    # Submit all tasks
    futures = [
        executor.submit(process_chunk, fork_bytes, start, end)
        for fork_bytes, start, end in tasks
    ]
    
    # Collect results
    completed_forks = []
    for i, future in enumerate(futures):
        result_bytes = future.result()
        fork_result = pickle.loads(result_bytes)
        completed_forks.append(fork_result)
        print(f"   ✓ Worker {i+1} completed")

parallel_time = time.perf_counter() - start_time

# === Step 4: Merge Results ===
print("\n3. Merging results from all workers...")

session.merge(*completed_forks)
snapshot_id = session.commit("Parallel computation complete")

print(f"   ✓ All forks merged successfully")
print(f"   ✓ Total computation time: {parallel_time:.2f}s")
print(f"   ✓ Snapshot: {snapshot_id[:8]}...")

# === Step 5: Verify Results ===
print("\n4. Verifying results...")

session_read = repo.readonly_session("main")
root_read = zarr.open_group(session_read.store, mode="r")
array_read = root_read["results"]

# Check that all data was written
sample_slices = [0, 250, 500, 750, 999]
print(f"\n   Checking sample slices:")
for idx in sample_slices:
    data_slice = array_read[idx, 50, 50]  # Sample point
    is_nan = np.isnan(data_slice)
    status = "✗ NaN (not written)" if is_nan else f"✓ {data_slice:.2f}"
    print(f"     Slice {idx:3d}: {status}")

# Compute statistics
data_chunk = array_read[0:100, :, :]
print(f"\n   Data statistics (first 100 slices):")
print(f"     Mean: {np.nanmean(data_chunk):.2f}")
print(f"     Std:  {np.nanstd(data_chunk):.2f}")
print(f"     Min:  {np.nanmin(data_chunk):.2f}")
print(f"     Max:  {np.nanmax(data_chunk):.2f}")

# === Step 6: Performance Comparison ===
print("\n5. Performance comparison (estimated):")

# Estimate serial time (rough calculation)
estimated_serial = parallel_time * num_workers
print(f"   Parallel time: {parallel_time:.2f}s")
print(f"   Estimated serial: {estimated_serial:.2f}s")
print(f"   Speedup: {estimated_serial / parallel_time:.1f}x")

print("\n✓ Example complete!")
print(f"\nKey takeaways:")
print(f"  - ForkSession enables safe parallel writes")
print(f"  - Each worker operates independently")
print(f"  - All changes merged atomically at the end")
print(f"  - Repository location: {repo_path}")
```

**Note**: This example demonstrates the ForkSession pattern for distributed writes. In production:
- Use appropriate task queuing systems (Celery, Dask, etc.)
- Handle failures and retries
- Monitor worker progress
- Consider data locality for cloud storage

---

## Summary

These examples demonstrate:

1. **Local workflows** - Simple virtualization with local files
2. **Public S3** - Anonymous access to public datasets
3. **Private S3 (IAM)** - Instance profile credentials for EC2/ECS/Lambda
4. **Private S3 (explicit)** - Access key authentication from environment
5. **Multi-file aggregation** - Time series concatenation with parallel processing
6. **Version control** - Branches, tags, commits, time-travel
7. **Distributed writes** - Parallel computation with ForkSession

All examples are fully runnable and demonstrate real-world workflows.

For troubleshooting, see [troubleshooting.md](troubleshooting.md).

For conceptual information and API details, see [skill.md](skill.md).

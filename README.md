# Virtual Zarr AI Agent Skill

AI agent skill for creating and working with virtual Zarr stores using VirtualiZarr and icechunk.

## What This Skill Does

Enables AI agents to help users virtualize legacy archival data formats (NetCDF, HDF5, GRIB, FITS) into cloud-optimized Zarr stores without copying the underlying data.

### Key Capabilities

- **Virtualization**: Convert NetCDF/HDF5 files to virtual Zarr references without data duplication
- **Multi-file aggregation**: Combine time series or ensemble datasets into unified virtual stores
- **Storage backends**: Support for S3 (public, IAM role, explicit credentials), GCS, Azure, and local filesystems
- **Version control**: Git-like operations (branches, tags, snapshots, time-travel) on array data via icechunk
- **Performance optimization**: Parallel processing, caching, and batch operations for large file collections

### Technologies

- **VirtualiZarr**: Creates lightweight manifests referencing chunks in existing files
- **Icechunk**: Provides transactional storage with ACID guarantees and version control for Zarr data

## When to Use This Skill

Activate this skill when working with:

- Virtual zarr, virtualizarr, or icechunk libraries
- Converting legacy archival formats to cloud-optimized formats
- Creating kerchunk references (legacy format)
- Building virtual datasets from collections of NetCDF/HDF5 files
- Setting up version-controlled array storage
- Virtualizing netcdf, hdf5, grib, or fits files
- Zarr references and manifest stores

## Prerequisites

- **Python**: 3.10 or higher
- **Libraries**: 
  - `virtualizarr` (latest stable)
  - `icechunk` (latest stable)
  - `xarray`, `zarr`, `obstore`, `obspec-utils`
  - Optional: `dask` for parallel operations

Install with:
```bash
pip install virtualizarr icechunk xarray zarr obstore obspec-utils
```

## Getting Started

1. **Start with skill.md** for comprehensive operational instructions and common workflows
2. **See examples.md** for complete runnable code examples covering:
   - Local file virtualization
   - Public S3 datasets (anonymous access)
   - Private S3 with IAM roles (EC2/ECS/Lambda)
   - Private S3 with explicit credentials
   - Multi-file time series aggregation
   - Icechunk version control workflows
   - Distributed parallel writes
3. **Refer to troubleshooting.md** when encountering errors or issues

## Quick Example

```python
import icechunk
import virtualizarr as vz
from obstore.store import from_url
from obspec_utils.registry import ObjectStoreRegistry
from virtualizarr.parsers import HDFParser

# Setup object store for source data
bucket = "s3://noaa-goes16"
store = from_url(bucket, region="us-east-1", skip_signature=True)
registry = ObjectStoreRegistry({bucket: store})

# Virtualize a file
url = f"{bucket}/ABI-L2-MCMIPF/2024/099/18/OR_ABI-L2-MCMIPF-M6_G16_s20240991800204_e20240991809524_c20240991810005.nc"
parser = HDFParser()
vds = vz.open_virtual_dataset(url, registry=registry, parser=parser, loadable_variables=['time', 'x', 'y'])

# Write to icechunk
storage = icechunk.local_filesystem_storage("/tmp/my-repo")
repo = icechunk.Repository.create(storage)
with repo.transaction("main", message="Initial virtualization") as store:
    vds.vz.to_icechunk(store)
```

## File Structure

```
virtual-zarr/
├── README.md             # This file - overview and quick start
├── skill.md              # Main operational instructions and workflows (~450 lines)
├── troubleshooting.md    # Detailed error scenarios and solutions (~250 lines)
└── examples.md           # Complete runnable examples (~400 lines)
```

## Skill Activation Keywords

This skill responds to:
- `virtual zarr`, `virtualizarr`
- `icechunk`
- `virtualize netcdf`, `virtualize hdf5`, `virtualize grib`, `virtualize fits`
- `kerchunk`
- `zarr references`, `manifest store`

## Key Concepts

- **Virtual References**: Lightweight metadata pointing to chunks in existing files (no data copying)
- **Loadable Variables**: Small arrays loaded into memory (coordinates, metadata)
- **Chunk Manifests**: Mappings of chunk keys to file paths, offsets, and lengths
- **Object Store Registry**: Maps URL prefixes to storage backends
- **Homogeneity Requirements**: Files must have matching chunk shapes, codecs, and data types

## Common Use Cases

### Scientific Data Management
- Weather forecasting and climate modeling
- Geospatial analysis and remote sensing
- Astronomical data processing

### AI/ML Workflows
- Training data versioning
- Model input reproducibility
- Large-scale dataset aggregation

### Collaborative Analysis
- Multiple teams accessing shared datasets safely
- Version-controlled data pipelines
- Audit trails for data transformations

### Archive Integration
- Accessing legacy HDF5/NetCDF without migration
- Cloud-optimized access to on-premises archives
- Gradual modernization of data infrastructure

## Benefits

- **No Data Duplication**: Virtual references point to existing files
- **Fast Aggregation**: Combine thousands of files in seconds
- **Version Control**: Track changes over time with git-like operations
- **Cloud-Optimized**: Efficient parallel chunk access
- **Storage Agnostic**: Works with any S3-compatible storage, GCS, Azure, or local filesystem
- **Transactional**: ACID guarantees prevent data corruption
- **Scalable**: Handle millions of files efficiently

## External Documentation

- [VirtualiZarr Documentation](https://virtualizarr.readthedocs.io/)
- [Icechunk Documentation](https://icechunk.io/)
- [Zarr Specification](https://zarr-specs.readthedocs.io/)
- [Xarray Documentation](https://docs.xarray.dev/)

## Contributing

This skill is designed to be comprehensive but concise. If you encounter issues or have suggestions:

- For VirtualiZarr: https://github.com/zarr-developers/VirtualiZarr/issues
- For Icechunk: https://github.com/earth-mover/icechunk/issues

## License

This skill documentation follows the licenses of the underlying projects:
- VirtualiZarr: Apache 2.0
- Icechunk: Apache 2.0

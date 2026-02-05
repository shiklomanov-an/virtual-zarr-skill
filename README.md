# Virtual Zarr AI Agent Skill

AI [agent skill][agentskills-docs] for creating and working with virtual Zarr stores using [VirtualiZarr][virtualizarr-docs] and [icechunk][icechunk-docs].

To use, clone the repo and copy the `virtual-zarr` folder to `~/.agents/skills/virtual-zarr`.

If you use [`chezmoi`](https://www.chezmoi.io), you can add something like the following to your `.chezmoiexternal.toml` to do this automatically:

```toml
[".agents/skills/virtual-zarr"]
type = "archive"
url = "https://github.com/shiklomanov-an/virtual-zarr-skill/archive/main.tar.gz"
exact = true
refreshPeriod = "24h"
include = ["*/virtual-zarr/**"]
stripComponents = 2
```

[virtualizarr-docs]: https://virtualizarr.readthedocs.io/
[icechunk-docs]: https://icechunk.io/
[agentskills-docs]: https://agentskills.io

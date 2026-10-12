# pii_data

The dataset store for the face blur (PII) work. Every dataset, every labeling pass, every
training mix and every training arm is described here. The pixels and the checkpoints are
not: they live on OSS.

The rule: **git is the ledger, OSS is the byte store.** Three commands are the whole
workflow.

```
git clone git@github.com:esteban-we/pii_data.git
bash data/setup.sh
git pull        # and git push
```

1. **Clone** gives the ledger only: no images, no checkpoints, no rendered view manifests.
2. **`bash data/setup.sh`** fast-forwards the checkout, sets `core.hooksPath .githooks`,
   pulls every byte the index names (236 GB of images plus each arm's pick checkpoint,
   about 20 minutes at 250 MB/s) and renders every view. It is idempotent, and anything
   you pass it goes to the pull, so `bash data/setup.sh --view train_Z5` or `--arm armAF`
   takes one slice instead; the render always covers all views, since it reads the
   indexes and not the images.
3. **`git pull` and `git push`** from then on: the hooks setup.sh enabled bring down the
   bytes a pull's new rows name, and send up the bytes this box made before a push leaves.
   A push is refused if an index row names bytes that are on neither this disk nor OSS.


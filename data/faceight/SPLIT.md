# faceight split

One frozen 80/20 train/eval split over all 34,667 faceight episodes (`episodes.txt`), decided once at the episode level. Every annotation round (a = round 1, b = round 2, c = round 3, any future round) is a projection of it: a frame follows its episode's side, both eyes included. No episode is ever on both sides, in any dataset, so a model trained on the train slice of any round is never evaluated on frames of an episode it saw.

Why episode: frames of one episode are near-duplicates (same session, same chunk, seconds apart); a frame-level split would leak. Rounds 2 and 3 share 15,069 episodes and 63 percent of their episodes are faceight_a episodes, so per-round splits would leak too.

How the sides were assigned (`mining/faceight_master_split.py`, seed 19760703, issue PII-1212):
- Pinned: the 19,783 faceight_a episodes keep the side of `faceight_a_{train,eval}_episodes_v1.txt` (PII-193; models were already trained on its train half); the 42 faceback_45 episodes keep their faceback side.
- Free (14,876): seeded stratified draw by scene_category (the 11 env buckets of `mining/faceight_sample.py`), face-free bin and face-count band, then a deterministic hill-climb so that episodes, frames and face-bearing frames are each at 20 percent eval within 1 pt and every scene within 2 pt.

Achieved eval share (episodes / frames): master 20.0 / 20.0; faceight_a 20.0 / 20.0; faceight_b 20.0 / 19.9; faceight_c 19.8 / 19.7; per scene 19.6 to 20.1.

Files in `data/splits/`: `faceight_{train,eval}_episodes_v1.txt` (master, 27,734 / 6,933), `faceight_{a,b,c}_{train,eval}_episodes_v1.txt` (projections), one sorted episode_id per line. `data/episode_usage.csv` carries the role per episode. To split a new round: run the script, which reproduces the master lists byte for byte and projects them; never redraw.

# facedub_a (pii2 store, PII-1718)

Round-1 human face boxes over the facedub VST frames, 37568 frames from 33 sessions
(18829 left, 18739 right), 1024 x 1280.
Written by data/build_facedub_pii2.py; `verify` re-derives every file from the drop copies.

    frames.csv        image,session,chunk,eye,frame_idx,t_ms,size,md5,round,local_name,src_oss_key,oss_key
                      size and md5 are the OSS object size and ETag of src_oss_key (the Verdict
                      object); the images themselves live on fluence (PII-1717), not here
    split.csv         session,role; session level, 25 train / 8 eval, seed 10330
    boxes/v1/         the round-1 pass: frames.csv (image,reviewed), boxes.csv (pixel xyxy at one
                      decimal, sorted by image,x1,y1), job/output/ byte copies of the two drops
    boxes/rec1/       the recognizable subset of boxes/v1 (PII-1915): the 208 of 6,849 reviewed
                      boxes that at least 1 of the 3 facereview passes called recognizable, on
                      2,046 eval frames, plus votes.csv with every pass, reviewer and timestamp

Names: image = <session>_c<chunk>_<left|right>_f<frame_idx:06d>.jpg; local_name is the shang file
<session>_c<chunk>_w<left|right>_t<t_ms>.jpg with t_ms = rank * 1000 + 233 inside its
(session, chunk, view) group (the sampler took one frame a second; frame_idx steps by 30 with drift).
chunk runs 000..006.

Eval sessions (8):
    20260923_121432_PCLUDB
    20260923_121605_HVDBLU
    20260923_122330_GYCPWK
    20260923_125039_UHFRXB
    20260923_125346_RLRJYC
    20260923_130558_DNQHAL
    20260923_131944_ANQLHK
    20260923_132656_FSKCBQ

Manifests (in the pii repo, not here): training/manifests/facedub_a_train_labelv2.txt and
facedub_a_eval_labelv2.txt, header "# facedub_a/<image> 1024 1280", box lines x1 y1 x2 y2 plus 15
keypoint slots at -1.0, zero-box frames kept as a bare header (negatives).
